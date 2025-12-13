import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.feature_extraction.text import CountVectorizer

####### Helper Function to solve date extraction issues.

def extract_date_robust(text):
    month_map = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
        'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }

    def get_mon(s):
        """Convert month name/abbr to '01'-'12' string."""
        s = s.lower()
        for k, v in month_map.items():
            if k in s: return v
        return '01'

    def get_day(s):
        """Extract digits from a string (e.g., '21st' -> '21')."""
        m = re.search(r'\d+', s)
        return m.group() if m else '01'

    # --- Format Parsers ---
    def parse_dmy(s):
        """Format: 15 May 2020"""
        parts = s.split()
        d = get_day(parts[0])
        m = parts[1]
        y = parts[-1]
        return f"{y}-{get_mon(m)}-{int(d):02d}"

    def parse_mdy(s):
        """Format: May 21, 2020"""
        s = s.replace(',', '')
        parts = s.split()
        m = parts[0]
        d = get_day(parts[1])
        y = parts[-1]
        return f"{y}-{get_mon(m)}-{int(d):02d}"

    def parse_dm(s):
        """Format: 30 June (Year missing -> 2020)"""
        parts = s.split()
        d = get_day(parts[0])
        m = parts[1]
        return f"2020-{get_mon(m)}-{int(d):02d}"

    def parse_md(s):
        """Format: March 20 (Year missing -> 2020)"""
        parts = s.split()
        m = parts[0]
        d = get_day(parts[1])
        return f"2020-{get_mon(m)}-{int(d):02d}"

    def parse_num(s):
        """Format: 30.03.2020 or 2020-03-30"""
        s = re.sub(r'[./-]', ' ', s)
        parts = s.split()
        p1, p2, p3 = parts[0], parts[1], parts[2]

        # Logic to detect Year vs Day
        if len(p1) == 4:  # YYYY-MM-DD
            y, m, d = p1, p2, p3
        elif len(p3) == 4:  # DD-MM-YYYY
            y, m, d = p3, p2, p1
        else:  # Fallback for 2-digit years, assume DD-MM-YY
            y = p3 if len(p3) == 4 else '20' + p3
            m, d = p2, p1

        return f"{y}-{int(m):02d}-{int(d):02d}"

    if not isinstance(text, str): return None
    text = text.replace("On ", "")  # Clean preamble

    months = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

    # Regex Patterns (Order determines priority)
    # 1. DD Month YYYY (e.g., 15 May 2020)
    r_dmy = rf"(?P<dmy>\d{{1,2}}\s+{months}\w*\s+\d{{4}})"
    # 2. Month DD, YYYY (e.g., May 21, 2020)
    r_mdy = rf"(?P<mdy>{months}\w*\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{4}})"
    # 3. DD Month (e.g., 30 June) -> Implies 2020
    r_dm = rf"(?P<dm>\d{{1,2}}\s+{months}\w*)"
    # 4. Month DD (e.g., March 30) -> Implies 2020
    r_md = rf"(?P<md>{months}\w*\s+\d{{1,2}}(?:st|nd|rd|th)?)"
    # 5. Numerical (e.g., 30.03.2020)
    r_num = r"(?P<num>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"

    # Combine patterns
    master_pat = f"{r_dmy}|{r_mdy}|{r_dm}|{r_md}|{r_num}"

    # Find the FIRST match in the string
    match = re.search(master_pat, text, re.IGNORECASE)

    if match:
        try:
            if match.group('dmy'):
                return parse_dmy(match.group('dmy'))
            elif match.group('mdy'):
                return parse_mdy(match.group('mdy'))
            elif match.group('dm'):
                return parse_dm(match.group('dm'))
            elif match.group('md'):
                return parse_md(match.group('md'))
            elif match.group('num'):
                return parse_num(match.group('num'))
        except:
            return None
    return None

# --- Plotting Functions (Unchanged) ---

def plot_measure_distribution(df, output_path):
    plt.figure(figsize=(12, 8))
    type_counts = df['type'].value_counts()
    sns.barplot(x=type_counts.values, y=type_counts.index, palette="viridis")
    plt.title('Distribution of COVID-19 Measure Types')
    plt.xlabel('Count')
    plt.ylabel('Measure Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'measure_types_distribution.png'))
    plt.close()


def plot_description_length_box(df, output_path):
    df['desc_length'] = df['description'].str.len()
    order = df.groupby('type')['desc_length'].median().sort_values(ascending=False).index
    plt.figure(figsize=(14, 10))
    sns.boxplot(x='desc_length', y='type', data=df, order=order, palette="viridis")
    plt.title('Distribution of Description Length by Measure Type')
    plt.xlabel('Description Length (characters)')
    plt.ylabel('Measure Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'description_length_box.png'))
    plt.close()


def plot_timeline(df, output_path):
    daily_counts = df['date_obj'].value_counts().sort_index()
    plt.figure(figsize=(14, 6))
    daily_counts.plot(kind='line', marker='o', color='teal')
    plt.title('Timeline of COVID-19 Measures (Daily Frequency)')
    plt.xlabel('Date')
    plt.ylabel('Number of Measures Announced')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'measures_timeline.png'))
    plt.close()


def plot_top_bigrams(df, output_path):
    try:
        vec = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=15)
        bow = vec.fit_transform(df['description'].dropna())
        sum_words = bow.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        df_bigrams = pd.DataFrame(words_freq, columns=['Bigram', 'Frequency'])
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Frequency', y='Bigram', data=df_bigrams, palette='magma')
        plt.title('Top 15 Most Frequent Phrases (Bigrams)')
        plt.xlabel('Frequency')
        plt.ylabel('Phrase')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'top_bigrams.png'))
        plt.close()
    except Exception as e:
        print(f"Skipping Bigram plot due to error: {e}")


def plot_description_length_violin(df, output_path):
    df['desc_length'] = df['description'].str.len()
    order = df.groupby('type')['desc_length'].median().sort_values(ascending=False).index
    plt.figure(figsize=(14, 10))
    sns.violinplot(x='desc_length', y='type', data=df, order=order, palette="coolwarm", inner="quartile")
    plt.title('Density Distribution of Description Length by Measure Type')
    plt.xlabel('Description Length (characters)')
    plt.ylabel('Measure Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'description_length_violin.png'))
    plt.close()


def plot_cumulative_growth(df, output_path):
    top_5_types = df['type'].value_counts().nlargest(5).index.tolist()
    df_pivot = df[df['type'].isin(top_5_types)].pivot_table(index='date_obj', columns='type', values='description',
                                                            aggfunc='count', fill_value=0)
    df_cumulative = df_pivot.cumsum()
    plt.figure(figsize=(14, 8))
    colors = sns.color_palette("husl", 5)
    for i, col in enumerate(df_cumulative.columns):
        plt.plot(df_cumulative.index, df_cumulative[col], label=col, linewidth=2, color=colors[i])
    plt.title('Cumulative Growth of Top 5 COVID-19 Measure Categories')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Number of Measures')
    plt.legend(title='Measure Type')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_path, 'cumulative_trends.png'))
    plt.close()


def plot_heatmap(df, output_path):
    df['Month'] = df['date_obj'].dt.month_name()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December']
    available_months = [m for m in month_order if m in df['Month'].unique()]
    if not available_months: return
    heatmap_data = pd.crosstab(df['type'], df['Month'])
    heatmap_data = heatmap_data[available_months]
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5)
    plt.title('Heatmap of Measure Frequency by Month and Type')
    plt.xlabel('Month')
    plt.ylabel('Measure Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'measures_heatmap.png'))
    plt.close()


def plot_day_of_week(df, output_path):
    df['Day_of_Week'] = df['date_obj'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['Day_of_Week'].value_counts().reindex(days_order)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=day_counts.index, y=day_counts.values, palette="Blues_d")
    plt.title('Frequency of Measures by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Measures')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_path, 'day_of_week_activity.png'))
    plt.close()


# --- Improved Date Parsing ---

def preprocess_dates(df):
    """
    Performs necessary preprocessing on the dataset before passing to the plots.
    """

    df['date_obj'] = df['description'].apply(extract_date_robust)

    # Filter valid dates
    initial_count = len(df)
    df_valid = df.dropna(subset=['date_obj'])
    parsed_count = len(df_valid)

    print(f"Dates successfully parsed: {parsed_count} out of {initial_count}")

    df_valid['date_obj'] = pd.to_datetime(
        df_valid['date_obj'],
        errors='coerce'
    )

    if parsed_count > 0:
        # Filter for relevant years (2020-2022) to remove OCR noise like '1900'
        df_valid = df_valid[(df_valid['date_obj'].dt.year >= 2020) & (df_valid['date_obj'].dt.year <= 2022)]

    return df_valid


def main(df, output_dir):
    print(f"Starting EDA generation. Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Generate content-based plots
    plot_measure_distribution(df, output_dir)
    plot_description_length_box(df, output_dir)
    plot_top_bigrams(df, output_dir)
    plot_description_length_violin(df, output_dir)

    # Generate time-based plots
    df_time = preprocess_dates(df)

    if not df_time.empty:
        plot_timeline(df_time, output_dir)
        plot_cumulative_growth(df_time, output_dir)
        plot_heatmap(df_time, output_dir)
        plot_day_of_week(df_time, output_dir)
    else:
        print("Warning: Could not parse any dates. Skipping time-based plots.")

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    DEFAULT_DATA = '../Dataset/dataset_raw.csv'
    DEFAULT_OUT = '../Visualizations/'
    if os.path.exists(DEFAULT_DATA):
        from data_preprocessing import load_and_augment_data_for_visualization

        df = load_and_augment_data_for_visualization(DEFAULT_DATA)
        main(df, DEFAULT_OUT)