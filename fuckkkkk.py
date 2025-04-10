df = pd.read_csv(csv_path, sep=',', usecols=['filename', 'words'], keep_default_na=False)
print(df.head())  # Check the first few rows