"""
Clinic <-> Lab Reconciliation
--------------------------------
Streamlit port of the original reconciliation script.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import re
import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz, process
import pdfplumber
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)

st.set_page_config(page_title="Clinic \u2194 Lab Reconciliation", layout="wide")

# --------------------------------------------------------------------------
# Regex / constants
# --------------------------------------------------------------------------
LAB_NO_RE = re.compile(r'^[A-Z]{2,4}\d{6,10}$')
DATE_RE = re.compile(r'^\d{2}\.\d{2}\.\d{4}$')
HVA_RE = re.compile(r'^[A-Z]{2}-\d{4,8}-')
SUFFIX_RE = re.compile(r'^[A-Z]{2,4}\d{4,9}$')

DEFAULT_UNWANTED_KEYWORDS = [
    'Labs', 'LABS', 'Registration', 'Jalan', 'Iskandar', '560 1042', '.com', 'INVOICE', 'SST No', 'Bill To',
    'JALAN', 'TAMAN', 'Date : ', 'Account No', 'Page', 'Reg No', 'Discount', 'MYR', 'Gross Tax', 'invoice',
    'Service Tax', 'No SST', 'Thank You', 'interest charge', 'considered', 'Bank', 'Swift code', 'prohibited',
    'Computer generated', 'Total Amount',
]

DEFAULT_TEST_RENAME = {'PAP SMEAR': 'H408'}
DEFAULT_EXCLUDE_TESTS = ['F/UP URINE']


# --------------------------------------------------------------------------
# Text cleaning / low-level parsing helpers (ported 1:1 from the script)
# --------------------------------------------------------------------------
def clean_lines(raw_lines, unwanted_keywords, unwanted_lines=None):
    unwanted_lines = unwanted_lines or []
    cleaned = [
        line for line in raw_lines
        if line.strip()
        and line.strip() not in unwanted_lines
        and not any(keyword in line for keyword in unwanted_keywords)
    ]
    return cleaned


def extract_total_amount_line(line):
    """
    Extract gross, discount, tax, and total from a line like:
    'Total Amount 33,901.70 0.00 0.00 33,901.70'
    """
    match = re.search(
        r'Total Amount\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})',
        line
    )
    if not match:
        return None
    gross, discount, tax, total = match.groups()
    return {
        'gross_amount': float(gross.replace(',', '')),
        'discount': float(discount.replace(',', '')),
        'tax_amount': float(tax.replace(',', '')),
        'total_amount': float(total.replace(',', '')),
    }


def extract_total_payable_line(line):
    """
    Extract total payable amount from a line like:
    'Computer generated document. No signature is required. Total Payable Amount 33,901.70'
    """
    match = re.search(r'Total Payable Amount\s+([\d,]+\.\d{2})', line)
    if not match:
        return None
    return float(match.group(1).replace(',', ''))


def is_lab_no(tok):
    return bool(LAB_NO_RE.match(tok))


def is_date(tok):
    return bool(DATE_RE.match(tok))


def looks_like_hva(tok):
    return bool(HVA_RE.match(tok))


def is_hva_suffix(tok):
    return bool(SUFFIX_RE.match(tok))


def is_probable_name_word(tok):
    return tok == '@' or tok.isalpha()


def is_id_no(tok):
    if not tok.isalnum():
        return False
    if tok.isdigit():
        return len(tok) == 12
    return 6 <= len(tok) <= 12 and any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)


def is_record_start(tokens, i):
    return i + 1 < len(tokens) and is_lab_no(tokens[i]) and is_id_no(tokens[i + 1])


def tokenize_lines(lines):
    tokens = []
    for line in lines:
        line = line.strip()
        if line:
            tokens.extend(line.split())
    return tokens


def parse_records(lines):
    tokens = tokenize_lines(lines)
    n = len(tokens)
    records, unresolved = [], []
    i = 0
    while i < n:
        tok = tokens[i]
        # 1. complete a dangling hva_no suffix on the previous record
        if records and records[-1]['hva_no'] and records[-1]['hva_no'].endswith('-') and is_hva_suffix(tok):
            records[-1]['hva_no'] += tok
            i += 1
            continue
        # 2. start a new record
        if is_record_start(tokens, i):
            lab_no, id_no = tokens[i], tokens[i + 1]
            j = i + 2
            hva_no = None
            if j < n and looks_like_hva(tokens[j]):
                hva_no = tokens[j]
                j += 1
            name_tokens = []
            while j < n and not is_date(tokens[j]):
                if is_record_start(tokens, j):
                    break
                name_tokens.append(tokens[j])
                j += 1
            if j >= n or not is_date(tokens[j]):
                unresolved.extend(tokens[i:j + 1] if j < n else tokens[i:j])
                i += 1
                continue
            try:
                date = tokens[j]; j += 1
                test = tokens[j]; j += 1
                qty = tokens[j]; j += 1
                unit_price = tokens[j]; j += 1
                gross_amount = tokens[j]; j += 1
                discount = tokens[j]; j += 1
                tax_code = tokens[j]; j += 1
                tax_amount = tokens[j]; j += 1
                total_amount = tokens[j]; j += 1
            except IndexError:
                unresolved.append(lab_no)
                i += 1
                continue
            records.append({
                'lab_no': lab_no, 'id_no': id_no, 'hva_no': hva_no,
                'inv_name': ' '.join(name_tokens), 'date': date, 'test': test,
                'qty': qty, 'unit_price': unit_price, 'gross_amount': gross_amount,
                'discount': discount, 'tax_code': tax_code,
                'tax_amount': tax_amount, 'total_amount': total_amount,
            })
            i = j
            continue
        # 3. stray token: word-shaped => wrapped name overflow, else truly orphaned
        if records and is_probable_name_word(tok):
            records[-1]['inv_name'] += ' ' + tok
        else:
            unresolved.append(tok)
        i += 1
    return records, unresolved


def build_dataframe(lines):
    """Returns (dataframe, list_of_warning_strings)."""
    records, unresolved = parse_records(lines)
    warnings = []
    df = pd.DataFrame(records)
    if df.empty:
        warnings.append("No lab line items could be parsed from this PDF.")
        return df, warnings
    for c in ['unit_price', 'gross_amount', 'discount', 'tax_amount', 'total_amount']:
        df[c] = df[c].str.replace(',', '').astype(float)
    df['qty'] = df['qty'].astype(int)
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    dangling = df[df['hva_no'].astype(str).str.endswith('-', na=False)]
    if not dangling.empty:
        warnings.append(
            f"{len(dangling)} records still have an incomplete hva_no: "
            + ", ".join(dangling['lab_no'].tolist())
        )
    if unresolved:
        warnings.append(f"{len(unresolved)} tokens unresolved: {unresolved}")
    return df, warnings


# --------------------------------------------------------------------------
# File-level parsers (adapted for Streamlit uploads / in-memory buffers)
# --------------------------------------------------------------------------
def parse_invoice_pdf(file_obj, password, unwanted_keywords):
    lines = []
    totals = {}
    with pdfplumber.open(file_obj, password=password) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=0.25)
            if not text:
                continue
            for line in text.split('\n'):
                stripped = line.strip()
                if line.startswith('Total Amount'):
                    parsed = extract_total_amount_line(line)
                    if parsed:
                        totals.update(parsed)
                elif 'Total Payable Amount' in line:
                    payable = extract_total_payable_line(line)
                    if payable is not None:
                        totals['total_payable_amount'] = payable
                lines.append(stripped)

    cleaned_lines = clean_lines(lines, unwanted_keywords)
    inv_data, warnings = build_dataframe(cleaned_lines)
    inv_data = inv_data.rename(columns={'date': 'lab_collected_date', 'id_no': 'lab_id_no', 'inv_name': 'lab_name'})
    if inv_data.empty:
        return inv_data, totals, warnings

    inv_data['lab_name'] = inv_data['lab_name'].str.strip().str.upper()
    inv_data['test'] = inv_data['test'].str.strip().str.upper()
    for title in ['MR ', 'MS ', 'MRS ']:
        inv_data['lab_name'] = inv_data['lab_name'].str.replace(title, '')
    inv_data['inv_seq'] = range(1, inv_data.shape[0] + 1)
    return inv_data, totals, warnings


def parse_lab_csv(file_obj, test_rename, exclude_tests):
    lab_data = pd.read_csv(file_obj)
    lab_data.columns = [col.lower().strip().replace(' ', '_').replace('.', '') for col in lab_data.columns]

    required = {'date', 'name', 'ic', 'package'}
    missing = required - set(lab_data.columns)
    if missing:
        raise ValueError(
            f"Lab CSV is missing expected column(s): {', '.join(sorted(missing))}. "
            f"Found columns: {', '.join(lab_data.columns)}"
        )

    lab_data = lab_data[['date', 'name', 'ic', 'package']]
    lab_data.columns = ['clinic_collected_date', 'clinic_name', 'clinic_id_no', 'clinic_test']

    lab_data['clinic_name'] = lab_data['clinic_name'].str.strip()
    lab_data['clinic_test'] = lab_data['clinic_test'].str.strip().str.upper()
    lab_data['clinic_collected_date'] = pd.to_datetime(lab_data['clinic_collected_date'], dayfirst=True)

    lab_data = lab_data.assign(clinic_test=lab_data['clinic_test'].str.split(',')).explode('clinic_test')
    lab_data['clinic_test'] = lab_data['clinic_test'].str.strip()
    lab_data = lab_data.reset_index(drop=True)

    rename_idx = lab_data[lab_data['clinic_test'].isin(test_rename.keys())].index
    lab_data.loc[rename_idx, 'clinic_test'] = lab_data.loc[rename_idx, 'clinic_test'].map(test_rename)

    lab_data['lab_seq'] = range(1, lab_data.shape[0] + 1)
    lab_data = lab_data[~lab_data['clinic_test'].isin(exclude_tests)]
    return lab_data


def strip_z(test):
    t = str(test).strip().upper()
    return t[1:] if t.startswith('Z') and len(t) > 1 else t


def build_name_mapping(lab_data, inv_data, score_cutoff=85):
    mapping = {}
    unmatched = []
    inv_by_id = inv_data.groupby('lab_id_no')['lab_name'].unique()
    for id_no, clinic_name in lab_data[['clinic_id_no', 'clinic_name']].drop_duplicates().itertuples(index=False):
        if id_no not in inv_by_id.index:
            unmatched.append((id_no, clinic_name, None, 0))
            continue
        candidates = inv_by_id.loc[id_no]
        if clinic_name in candidates:
            continue  # exact match already, no remap needed
        best_match, score = process.extractOne(clinic_name, candidates, scorer=fuzz.token_sort_ratio)
        if score >= score_cutoff:
            mapping[clinic_name] = best_match
        else:
            unmatched.append((id_no, clinic_name, best_match, score))
    return mapping, unmatched


def build_id_mapping(lab_data, inv_data, score_cutoff=90):
    """
    Fuzzy-match clinic_id_no against lab_id_no for typos (transposed digits,
    a missing/extra character, etc.), anchored on the (already name-matched)
    person's name rather than the ID itself. Must run after name matching so
    that lab_data['name_matched'] is populated.
    """
    mapping = {}
    unmatched = []
    inv_by_name = inv_data.groupby('lab_name')['lab_id_no'].unique()
    for name, clinic_id in lab_data[['name_matched', 'clinic_id_no']].drop_duplicates().itertuples(index=False):
        if name not in inv_by_name.index:
            unmatched.append((name, clinic_id, None, 0))
            continue
        candidates = inv_by_name.loc[name]
        if clinic_id in candidates:
            continue  # exact match already, no remap needed
        best_match, score = process.extractOne(clinic_id, candidates, scorer=fuzz.ratio)
        if score >= score_cutoff:
            mapping[clinic_id] = best_match
        else:
            unmatched.append((name, clinic_id, best_match, score))
    return mapping, unmatched


def match_with_date_shift(lab_data, inv_data):
    lab = lab_data.copy()
    inv = inv_data.copy()
    lab['clinic_collected_date'] = pd.to_datetime(lab['clinic_collected_date'])
    inv['lab_collected_date'] = pd.to_datetime(inv['lab_collected_date'])

    # Stage 1: same-day outer match
    same_day = lab.merge(
        inv, left_on=['clinic_collected_date', 'name_matched', 'id_matched', 'clinic_test'],
        right_on=['lab_collected_date', 'lab_name', 'lab_id_no', 'test'],
        how='outer', indicator=True
    )
    matched = same_day[same_day['_merge'] == 'both'].drop(columns=['_merge'])
    matched['match_type'] = 'same_day'
    lab_only = same_day[same_day['_merge'] == 'left_only']['lab_seq'].dropna().unique()
    inv_only = same_day[same_day['_merge'] == 'right_only']['inv_seq'].dropna().unique()
    unmatched_lab = lab[lab['lab_seq'].isin(lab_only)].copy()
    unmatched_inv = inv[inv['inv_seq'].isin(inv_only)].copy()

    # Stage 2: next-day outer match (only on leftovers from stage 1)
    if len(unmatched_lab) and len(unmatched_inv):
        unmatched_lab['collect_date_shifted'] = unmatched_lab['clinic_collected_date'] + pd.Timedelta(days=1)
        next_day = unmatched_lab.merge(
            unmatched_inv, left_on=['collect_date_shifted', 'name_matched', 'id_matched', 'clinic_test'],
            right_on=['lab_collected_date', 'lab_name', 'lab_id_no', 'test'],
            how='outer', indicator=True
        )
        matched_next_day = next_day[next_day['_merge'] == 'both'].drop(columns=['_merge', 'collect_date_shifted'])
        matched_next_day['match_type'] = 'next_day'
        lab_only_2 = next_day[next_day['_merge'] == 'left_only']['lab_seq'].dropna().unique()
        inv_only_2 = next_day[next_day['_merge'] == 'right_only']['inv_seq'].dropna().unique()
        still_unmatched_lab = unmatched_lab[unmatched_lab['lab_seq'].isin(lab_only_2)].drop(columns=['collect_date_shifted'])
        still_unmatched_inv = unmatched_inv[unmatched_inv['inv_seq'].isin(inv_only_2)].copy()
    else:
        matched_next_day = lab.iloc[0:0]
        still_unmatched_lab = unmatched_lab
        still_unmatched_inv = unmatched_inv

    return matched, matched_next_day, still_unmatched_lab, still_unmatched_inv


def flag_z_mismatches(still_unmatched_lab, still_unmatched_inv):
    inv = still_unmatched_inv.copy()  # restricted to leftover lab rows only
    inv['test_stripped'] = inv['test'].apply(strip_z)
    lab = still_unmatched_lab.copy()
    lab['test_stripped'] = lab['clinic_test'].apply(strip_z)

    resolved_rows = []
    truly_unmatched_lab_rows = []
    tag_on_notes = []
    used_inv_seq = set()

    for _, row in lab.iterrows():
        candidates = inv[
            (inv['lab_name'] == row['name_matched']) &
            (inv['lab_id_no'] == row['id_matched']) &
            (inv['lab_collected_date'].isin([row['clinic_collected_date'], row['clinic_collected_date'] + pd.Timedelta(days=1)])) &
            (inv['test_stripped'] == row['test_stripped']) &
            (inv['test'].str.upper() != row['clinic_test'].strip().upper()) &
            (~inv['inv_seq'].isin(used_inv_seq))
        ]
        if candidates.empty:
            truly_unmatched_lab_rows.append(row)
            continue

        inv_row = candidates.iloc[0]
        used_inv_seq.add(inv_row['inv_seq'])
        lab_has_z = row['clinic_test'].strip().upper().startswith('Z')
        inv_has_z = inv_row['test'].strip().upper().startswith('Z')
        clinic_name, clinic_id_no, cdate = row['name_matched'], row['clinic_id_no'], row['clinic_collected_date'].date()

        if inv_has_z and not lab_has_z:
            billing_status = 'Under Billed'
        elif lab_has_z and not inv_has_z:
            billing_status = 'Over Billed'
        else:
            billing_status = 'Other Mismatch'

        if billing_status in ('Under Billed', 'Over Billed'):
            tag_on_notes.append({
                'name': clinic_name, 'clinic_id_no': clinic_id_no, 'clinic_collected_date': cdate,
                'clinic_test': row['clinic_test'], 'lab_test': inv_row['test'],
                'billing_status': billing_status, 'amount': inv_row['total_amount'],
            })

        merged_row = {**inv_row.to_dict(), **row.to_dict(), 'match_type': 'z_mismatch', 'billing_status': billing_status}
        resolved_rows.append(merged_row)

    resolved_df = pd.DataFrame(resolved_rows)
    truly_unmatched_lab_df = pd.DataFrame(truly_unmatched_lab_rows)
    remaining_inv_df = inv[~inv['inv_seq'].isin(used_inv_seq)].copy()
    return resolved_df, truly_unmatched_lab_df, remaining_inv_df, tag_on_notes


def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("\U0001F9EA Clinic \u2194 Lab Reconciliation")
st.caption(
    "Upload the lab's invoice PDF and the clinic's record CSV to reconcile the lab's billed "
    "tests against what the clinic collected, and flag mismatches or discrepancies."
)

with st.sidebar:
    st.header("Inputs")
    inv_file = st.file_uploader("Lab PDF", type=["pdf"])
    pdf_password = st.text_input(
        "PDF Password (leave blank if the PDF isn't protected)", value="", type="password"
    )
    lab_file = st.file_uploader("Clinic Record CSV", type=["csv"])

    with st.expander("Advanced settings"):
        score_cutoff = st.slider("Fuzzy name match cutoff", 50, 100, 85)
        id_score_cutoff = st.slider("Fuzzy ID match cutoff", 50, 100, 90)
        exclude_tests_text = st.text_input(
            "Exclude tests (comma separated)", value=", ".join(DEFAULT_EXCLUDE_TESTS)
        )
        exclude_tests = [t.strip().upper() for t in exclude_tests_text.split(',') if t.strip()]

        st.markdown("**Clinic test name rename**")
        st.caption("Rename a clinic test name to the code used on the lab invoice, e.g. 'PAP SMEAR' \u2192 'H408'.")
        test_rename_default = pd.DataFrame(
            [{"clinic_test_name": k, "rename_to": v} for k, v in DEFAULT_TEST_RENAME.items()]
        )
        test_rename_df = st.data_editor(
            test_rename_default, num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={
                "clinic_test_name": st.column_config.TextColumn("Clinic test name"),
                "rename_to": st.column_config.TextColumn("Rename to"),
            },
            key="test_rename_editor",
        )
        test_rename = {
            str(r["clinic_test_name"]).strip().upper(): str(r["rename_to"]).strip()
            for _, r in test_rename_df.iterrows()
            if pd.notna(r["clinic_test_name"]) and str(r["clinic_test_name"]).strip()
            and pd.notna(r["rename_to"]) and str(r["rename_to"]).strip()
        }

    run_btn = st.button("Run Reconciliation", type="primary", disabled=not (inv_file and lab_file))

if 'results' not in st.session_state:
    st.session_state.results = None

if run_btn:
    try:
        with st.spinner("Parsing lab PDF..."):
            inv_bytes = io.BytesIO(inv_file.getvalue())
            inv_data, totals, inv_warnings = parse_invoice_pdf(
                inv_bytes, pdf_password, DEFAULT_UNWANTED_KEYWORDS
            )
        if inv_data.empty:
            st.error("No lab line items were extracted. Check the PDF password and file.")
            st.stop()

        with st.spinner("Parsing clinic record CSV..."):
            lab_bytes = io.BytesIO(lab_file.getvalue())
            lab_data = parse_lab_csv(lab_bytes, test_rename, exclude_tests)

        with st.spinner("Matching clinic records to lab lines..."):
            name_mapping, name_unmatched = build_name_mapping(lab_data, inv_data, score_cutoff=score_cutoff)
            lab_data['name_matched'] = lab_data['clinic_name'].replace(name_mapping)
            inv_data = inv_data.reset_index(drop=True)
            inv_data['inv_seq'] = inv_data.index

            id_mapping, id_unmatched = build_id_mapping(lab_data, inv_data, score_cutoff=id_score_cutoff)
            lab_data['id_matched'] = lab_data['clinic_id_no'].replace(id_mapping)

            matched_same_day, matched_next_day, still_unmatched_lab, still_unmatched_inv = match_with_date_shift(
                lab_data, inv_data
            )
            z_resolved, truly_unmatched_lab, truly_unmatched_inv, tag_on_notes = flag_z_mismatches(
                still_unmatched_lab, still_unmatched_inv
            )
            check_data = pd.concat([matched_same_day, matched_next_day, z_resolved], ignore_index=True)

            # Merge matched lab amounts back onto the full clinic dataset (by lab_seq),
            # so we can check whether the amounts *lab_data* (clinic side) reconcile to the lab total.
            lab_data_with_amount = lab_data.merge(
                check_data[['lab_seq', 'total_amount']], on='lab_seq', how='left'
            )

        inv_correct_total = totals.get('gross_amount')
        inv_current_total = float(np.round(inv_data['total_amount'].sum(), 2))
        totals_match = (inv_correct_total is not None) and (inv_correct_total == inv_current_total)

        lab_matched_total = float(np.round(lab_data_with_amount['total_amount'].sum(skipna=True), 2))
        lab_totals_match = (inv_correct_total is not None) and (inv_correct_total == lab_matched_total)

        st.session_state.results = dict(
            inv_data=inv_data, lab_data=lab_data, totals=totals, inv_warnings=inv_warnings,
            name_mapping=name_mapping, name_unmatched=name_unmatched,
            id_mapping=id_mapping, id_unmatched=id_unmatched,
            matched_same_day=matched_same_day, matched_next_day=matched_next_day,
            z_resolved=z_resolved, tag_on_notes=tag_on_notes,
            truly_unmatched_lab=truly_unmatched_lab, truly_unmatched_inv=truly_unmatched_inv,
            check_data=check_data, lab_data_with_amount=lab_data_with_amount,
            inv_correct_total=inv_correct_total, inv_current_total=inv_current_total,
            totals_match=totals_match,
            lab_matched_total=lab_matched_total, lab_totals_match=lab_totals_match,
        )
    except Exception as e:
        st.error(f"Reconciliation failed: {e}")
        st.exception(e)
        st.stop()

res = st.session_state.results

if res is None:
    st.info("Upload the lab PDF and clinic CSV in the sidebar, then click **Run Reconciliation**.")
else:
    for w in res['inv_warnings']:
        st.warning(w)

    total_lab = len(res['lab_data'])
    total_inv = len(res['inv_data'])
    n_same_day = len(res['matched_same_day'])
    n_next_day = len(res['matched_next_day'])
    n_z = len(res['z_resolved'])
    n_unmatched_lab = len(res['truly_unmatched_lab'])
    n_unmatched_inv = len(res['truly_unmatched_inv'])

    tabs = st.tabs([
        "Summary", "Matched", "Tag-On Mismatches", "Unmatched \u2013 Clinic", "Unmatched \u2013 Lab",
        "Name Typo", "ID Typo", "Raw Data",
    ])

    # ------------------------------------------------------------------
    # Tab 0: Summary
    # ------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Match Outcome")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Same-day Matches", n_same_day)
        c2.metric("Next-day Matches", n_next_day)
        c3.metric("Tag-On Mismatches", n_z)
        c4.metric("Unmatched (Clinic)", n_unmatched_lab, delta=f"of {total_lab} clinic records", delta_color="off")
        c5.metric("Unmatched (Lab)", n_unmatched_inv, delta=f"of {total_inv} lab lines", delta_color="off")

        match_summary_df = pd.DataFrame([
            {"category": "Same-day Match", "count": n_same_day},
            {"category": "Next-day Match", "count": n_next_day},
            {"category": "Tag-On Mismatch", "count": n_z},
            {"category": "Unmatched (Clinic)", "count": n_unmatched_lab},
            {"category": "Unmatched (Lab)", "count": n_unmatched_inv},
        ])
        match_chart = (
            alt.Chart(match_summary_df)
            .mark_bar(color="#8ecae6")
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("category:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260, labelFontSize=13)),
                tooltip=["category", "count"],
            )
            .properties(height=220)
        )
        st.altair_chart(match_chart, use_container_width=True)

        st.divider()

        st.subheader("Unmatched Records Per Day")
        lab_df = res['truly_unmatched_lab']
        inv_df = res['truly_unmatched_inv']

        if lab_df.empty and inv_df.empty:
            st.success("Every clinic record and every lab line was matched \u2014 no unmatched records on any day.")
        else:
            lab_by_day = (
                lab_df.assign(date=lab_df['clinic_collected_date'].dt.date)
                .groupby('date').size().rename('unmatched_clinic_count')
                if not lab_df.empty else pd.Series(name='unmatched_clinic_count', dtype=int)
            )
            inv_by_day = (
                inv_df.assign(date=inv_df['lab_collected_date'].dt.date)
                .groupby('date').size().rename('unmatched_lab_count')
                if not inv_df.empty else pd.Series(name='unmatched_lab_count', dtype=int)
            )
            by_day = pd.concat([lab_by_day, inv_by_day], axis=1).fillna(0).astype(int)
            by_day = by_day.sort_index().reset_index().rename(columns={'index': 'date'})
            st.dataframe(by_day, use_container_width=True, hide_index=True)
            st.download_button(
                "Download unmatched-per-day summary (CSV)", to_csv_bytes(by_day),
                file_name="unmatched_by_day.csv", mime="text/csv"
            )

        st.divider()

        st.subheader("Total amount checks")
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Check 1: extracted lab line items vs. lab PDF total**")
            st.metric(
                "Lab total (from PDF)",
                f"RM {res['inv_correct_total']:,.2f}" if res['inv_correct_total'] is not None else "N/A",
            )
            st.metric("Sum of parsed lab line items", f"RM {res['inv_current_total']:,.2f}")
            if res['totals_match']:
                st.success("Matches \u2014 all lab lines were correctly extracted.")
            else:
                st.error("Does NOT match \u2014 possible extraction error in the lab PDF.")
        with tc2:
            st.markdown("**Check 2: clinic_data (matched amounts) vs. lab PDF total**")
            st.metric(
                "Lab total (from PDF)",
                f"RM {res['inv_correct_total']:,.2f}" if res['inv_correct_total'] is not None else "N/A",
            )
            st.metric("Sum of lab amounts merged onto clinic_data", f"RM {res['lab_matched_total']:,.2f}")
            if res['lab_totals_match']:
                st.success("Matches \u2014 clinic_data fully accounts for the lab total.")
            else:
                st.error("Does NOT match \u2014 some clinic records are unmatched or misbilled; see the unmatched tabs.")

    # ------------------------------------------------------------------
    # Tab 1: Matched
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Matched records (same-day + next-day)")
        matched_all = pd.concat([res['matched_same_day'], res['matched_next_day']], ignore_index=True)
        st.dataframe(matched_all, use_container_width=True)
        st.download_button(
            "Download matched records (CSV)", to_csv_bytes(matched_all),
            file_name="matched_records.csv", mime="text/csv"
        )

    # ------------------------------------------------------------------
    # Tab 2: Tag-On Mismatches
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Tag-On Mismatches")
        if res['tag_on_notes']:
            st.dataframe(pd.DataFrame(res['tag_on_notes']), use_container_width=True)
        else:
            st.caption("No tag-on billing discrepancies detected.")
        st.subheader("Resolved Tag-On Mismatches")
        st.dataframe(res['z_resolved'], use_container_width=True)
        if not res['z_resolved'].empty:
            st.download_button(
                "Download resolved tag-on mismatches (CSV)", to_csv_bytes(res['z_resolved']),
                file_name="resolved_tag_on_mismatches.csv", mime="text/csv"
            )

    # ------------------------------------------------------------------
    # Tab 3: Unmatched - Clinic
    # ------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Clinic records with no matching lab line")
        df = res['truly_unmatched_lab']
        if df.empty:
            st.caption("None \u2014 every clinic record was matched.")
        else:
            for date, grp in df.sort_values('clinic_collected_date').groupby('clinic_collected_date'):
                st.markdown(f"**{date.date()}**")
                st.dataframe(grp[['name_matched', 'clinic_id_no', 'clinic_test']], use_container_width=True, hide_index=True)
            st.download_button(
                "Download unmatched clinic records (CSV)", to_csv_bytes(df),
                file_name="unmatched_clinic_records.csv", mime="text/csv"
            )

    # ------------------------------------------------------------------
    # Tab 4: Unmatched - Lab
    # ------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Lab lines with no matching clinic record")
        df = res['truly_unmatched_inv']
        if df.empty:
            st.caption("None \u2014 every lab line was matched.")
        else:
            for date, grp in df.sort_values('lab_collected_date').groupby('lab_collected_date'):
                st.markdown(f"**{date.date()}**")
                st.dataframe(grp[['lab_name', 'lab_id_no', 'test']], use_container_width=True, hide_index=True)
            st.download_button(
                "Download unmatched lab records (CSV)", to_csv_bytes(df),
                file_name="unmatched_lab_records.csv", mime="text/csv"
            )

    # ------------------------------------------------------------------
    # Tab 5: Name Typo
    # ------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Fuzzy name mapping applied (clinic name \u2192 lab name)")
        if res['name_mapping']:
            st.dataframe(
                pd.DataFrame(res['name_mapping'].items(), columns=['clinic_name', 'mapped_to_lab_name']),
                use_container_width=True,
            )
        else:
            st.caption("No fuzzy remapping was needed.")
        st.subheader("Names that could not be confidently matched")
        if res['name_unmatched']:
            st.dataframe(
                pd.DataFrame(res['name_unmatched'], columns=['clinic_id_no', 'clinic_name', 'closest_lab_name', 'score']),
                use_container_width=True,
            )
        else:
            st.caption("None.")

    # ------------------------------------------------------------------
    # Tab 6: ID Typo
    # ------------------------------------------------------------------
    with tabs[6]:
        st.subheader("Fuzzy ID mapping applied (clinic ID \u2192 lab ID)")
        st.caption("Matched using the person's name as an anchor, so this only catches ID typos where the name matched correctly.")
        if res['id_mapping']:
            st.dataframe(
                pd.DataFrame(res['id_mapping'].items(), columns=['clinic_id_no', 'mapped_to_lab_id_no']),
                use_container_width=True,
            )
        else:
            st.caption("No fuzzy ID remapping was needed.")
        st.subheader("IDs that could not be confidently matched")
        if res['id_unmatched']:
            st.dataframe(
                pd.DataFrame(res['id_unmatched'], columns=['name_matched', 'clinic_id_no', 'closest_lab_id_no', 'score']),
                use_container_width=True,
            )
        else:
            st.caption("None.")

    # ------------------------------------------------------------------
    # Tab 7: Raw Data
    # ------------------------------------------------------------------
    with tabs[7]:
        st.subheader("Parsed lab data")
        st.dataframe(res['inv_data'], use_container_width=True)
        st.subheader("Parsed clinic record data (with matched lab amount)")
        st.dataframe(res['lab_data_with_amount'], use_container_width=True)
