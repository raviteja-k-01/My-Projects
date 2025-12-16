# Uber Ride Cancellations & Wasted Capacity Analysis

## Project Overview
Ride-hailing platforms rarely fail because of pricing or lack of demand. They fail when availability breaks — when the right vehicle is not available at the right place and time. This project analyzes Uber ride booking data to identify where, when, and why availability failures occur, and how those failures translate into cancellations, wasted capacity, and revenue loss.

Instead of treating cancellations as a surface-level metric, the analysis connects cancellations to deeper operational gaps such as driver unavailability during peak hours, vehicle-type shortages, location-level demand–supply mismatch, and driver-initiated cancellations or “No Driver Found” scenarios. The objective is not to build solutions, but to diagnose the problem clearly enough for informed decision-making.

## Business Questions Addressed
- Is availability the primary driver behind ride cancellations?
- During which hours does availability break most frequently?
- Which vehicle types contribute the most to cancellations and revenue loss?
- Which locations consistently experience demand–supply imbalance?
- How much revenue and capacity is lost due to availability failures?

## Tech Stack
- SQL for data exploration, filtering, aggregation, and metric creation
- Python (Pandas, NumPy) for data cleaning, validation, and trend analysis
- Power BI for interactive dashboards and business storytelling

## Project Structure
Uber-Ride-Cancellations-and-Wasted-Capacity-Analysis/
├── data/
│   ├── raw/
│   └── clean/
├── notebook/
│   ├── 01_Data_Exploration&Querying.sql
│   └── 02_Python_Data_Analysis.ipynb
├── Dashboard/
│   └── Uber Ride Analytics.pbix
├── Report/
│   └── Uber Ride Analytics.pdf
└── README.md

## Dashboard Overview
The Power BI dashboard is structured into three analytical views.

Overview page presents a high-level snapshot of platform performance including total bookings, completed rides, cancellation rate, booking status distribution, and monthly booking and revenue trends. This establishes the scale and impact of cancellations before drilling into causes.

Cancellations analysis focuses on why rides fail by examining cancellations by vehicle type, peak cancellation hours, monthly trends, and location-level hotspots. The patterns clearly show that cancellations are concentrated during high-demand periods, indicating availability failure rather than lack of demand.

Wasted capacity and revenue impact quantifies the business cost of availability breakdown by measuring revenue loss by vehicle type, wasted capacity by cancellation reason, and peak hours of capacity loss. This translates operational issues into financial terms.

## Key Insights
- Availability is the primary constraint, not demand.
- Cancellations spike during peak demand hours, confirming supply-side failure.
- Driver-side cancellations contribute more to revenue loss than customer cancellations.
- Certain vehicle types consistently account for disproportionate revenue leakage.
- Revenue loss and wasted capacity follow repeatable, predictable patterns rather than random anomalies.

## Scope Clarification
This project diagnoses availability and cancellation drivers, quantifies wasted capacity and revenue loss, and identifies when and where operational intervention is required. It does not propose pricing strategies, incentive models, dispatch optimization, or algorithmic solutions. Those belong to product and operations teams. This work delivers the analytical foundation required to support those decisions.

## How to Clone and Run the Project
Clone the repository:
git clone https://github.com/raviteja-k-01/Uber-Ride-Cancellations-and-Wasted-Capacity-Analysis.git

Open the Power BI dashboard:
Navigate to the Dashboard folder and open Uber Ride Analytics.pbix using Power BI Desktop.

Optional analysis review:
SQL queries are available in notebook/01_Data_Exploration&Querying.sql and Python analysis is available in notebook/02_Python_Data_Analysis.ipynb.

## Author
Ravi Teja Kesagani  
GitHub: https://github.com/raviteja-k-01  
LinkedIn: https://www.linkedin.com/in/ravitejakesagani1
