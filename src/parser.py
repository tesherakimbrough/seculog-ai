import pandas as pd

def parse_logs(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.strip().lower() for col in df.columns]
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Simple detection: flag "denied" events
        df['suspicious'] = df['event'].str.lower().eq('denied')
        
        # Brute-force detection: flag if source_ip has >=3 denied events
        denied_counts = df[df['event'].str.lower() == 'denied']['source_ip'].value_counts()
        brute_force_ips = denied_counts[denied_counts >= 3].index.tolist()
        df['brute_force'] = df['source_ip'].isin(brute_force_ips) & df['event'].str.lower().eq('denied')
        
        return df
    except Exception as e:
        print(f"Error parsing logs: {e}")
        return pd.DataFrame()


