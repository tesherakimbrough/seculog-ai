from flask import Flask, render_template, request
from src.parser import parse_logs
import os
import plotly.graph_objs as go  # NEW
import plotly.io as pio         # NEW

app = Flask(__name__)

def make_event_chart(df):  # NEW
    # Bar chart: count of each event type
    event_counts = df['event'].value_counts()
    fig = go.Figure([go.Bar(x=event_counts.index, y=event_counts.values)])
    fig.update_layout(title='Event Type Counts', xaxis_title='Event', yaxis_title='Count')
    return pio.to_html(fig, full_html=False)

def make_top_ip_chart(df):  # NEW
    # Pie chart: top source_ip
    top_ips = df['source_ip'].value_counts().nlargest(5)
    fig = go.Figure([go.Pie(labels=top_ips.index, values=top_ips.values)])
    fig.update_layout(title='Top 5 Source IPs')
    return pio.to_html(fig, full_html=False)

@app.route("/", methods=["GET", "POST"])
def home():
    logs = None
    event_chart = None      # NEW
    top_ip_chart = None     # NEW
    if request.method == "POST":
        file = request.files["logfile"]
        if file:
            file_path = os.path.join("data", "uploaded.csv")
            file.save(file_path)
            logs = parse_logs(file_path)
            if logs is not None and not logs.empty:    # NEW
                event_chart = make_event_chart(logs)    # NEW
                top_ip_chart = make_top_ip_chart(logs)  # NEW
    return render_template("index.html", logs=logs, event_chart=event_chart, top_ip_chart=top_ip_chart)  # NEW

if __name__ == "__main__":
    app.run(debug=True)
