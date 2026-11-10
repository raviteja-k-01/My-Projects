# Retail Sales Insights & Customer Segmentation

## About This Project
I built this end-to-end analytics project to analyze retail sales data and uncover meaningful business insights. The project walks through the complete data analysis pipeline - from exploring and cleaning raw data with Python, running SQL queries for business metrics, to creating an interactive Power BI dashboard that anyone can understand.

## What I Set Out to Do
- Clean and explore messy retail sales data
- Find patterns in sales trends across different time periods
- Identify which countries generate the most revenue
- Group customers based on their buying behavior (RFM analysis)
- Build a dashboard that non-technical people can actually use
- Share insights that can drive business decisions

## Tech Stack
I used a combination of tools to complete this project:
- **Python** (Pandas, NumPy, Matplotlib) - for initial data exploration and cleaning in Google Colab
- **MySQL** - for storing data and running business queries
- **Power BI** - for building interactive visualizations
- **GitHub** - for organizing and sharing my work

## How Everything is Organized
```
Retail-Sales-Insights-Customer-Segmentation/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── data_exploration_cleaning.ipynb
│
├── data/
│   ├── raw/
│   │   └── online_retail.csv
│   ├── cleaned/
│   │   └── cleaned.csv
│   └── processed/
│       ├── total_sales.csv
│       ├── top_countries_sales.csv
│       ├── monthly_sales_trend.csv
│       └── rfm_scores.csv
│
├── sql/
│   ├── 01_create_database.sql
│   ├── 02_core_business_queries.sql
│   ├── 03_business_analysis.sql
│   ├── 04_rfm_analysis.sql
│   └── 05_kpi_queries.sql
│
├── powerbi/
│   ├── Retail_Sales_Dashboard.pbix
│   └── dashboard_screenshots/
│       ├── page1_overview.png
│       ├── page2_monthly_trend.png
│       ├── page3_top_countries.png
│       └── page4_customer_segmentation.png
│
├── reports/
│   └── project_report.pdf
│
└── images/
```

## The Dashboard
I created 4 pages in Power BI, each focusing on a different aspect of the business:

**Page 1 - Overview**  
Simple cards showing total revenue and transaction count. I wanted to give a quick snapshot of overall performance.

**Page 2 - Monthly Sales Trend**  
A line chart tracking how sales changed month by month. This helps spot seasonal patterns.

**Page 3 - Top Countries**  
Bar chart showing which countries bring in the most money. United Kingdom turned out to be the clear winner.

**Page 4 - Customer Segmentation**  
A donut chart breaking down customers into 5 groups based on how recently they bought, how often they shop, and how much they spend. This is super useful for targeted marketing.

## Key Findings
- Total Revenue: [Add your number]
- Total Transactions: [Add your number]
- Top performing country: United Kingdom
- Identified 5 distinct customer segments (Loyal, At Risk, New, Big Spenders, Regular)

## My Process

**Step 1: Cleaning the Data (Python)**  
Started by loading the raw CSV into Google Colab. Dealt with missing values, removed duplicates, fixed data types, and got rid of any weird outliers. Also created some quick plots to understand what I was working with.

**Step 2: Setting Up the Database (MySQL)**  
Created a database called `retail` and loaded my cleaned data. This made it way easier to run complex queries.

**Step 3: Running Business Queries (SQL)**  
Wrote queries to answer specific questions - like which products sell best, what days people shop most, how returns compare to sales, and of course the RFM analysis. Exported the results as CSVs for Power BI.

**Step 4: Building the Dashboard (Power BI)**  
Imported all my processed data, set up relationships between tables, and built out the 4 pages with different visuals. Added filters so users can slice the data however they want.

**Step 5: Documenting Everything**  
Organized all my files, wrote this README, and pushed everything to GitHub so others can learn from it or build on it.

## Want to Try It Yourself?

**You'll need:**
- Python 3.7 or higher
- MySQL
- Power BI Desktop (it's free)
- A GitHub account

**Steps:**
1. Clone this repo:
   ```bash
   git clone https://github.com/raviteja-k-01/Retail-Sales-Insights-Customer-Segmentation.git
   cd Retail-Sales-Insights-Customer-Segmentation
   ```

2. Check out the Python notebook in Google Colab to see how I cleaned the data

3. Run the SQL scripts in order (they're numbered) to recreate the database and queries

4. Open the Power BI file to explore the dashboard

## SQL Stuff I Did
- Counted records and checked data quality
- Calculated total revenue and transactions
- Ranked top countries by sales
- Analyzed monthly trends
- Found best-selling products
- Identified top-spending customers
- Looked at sales by day of week
- Compared quarterly performance
- Analyzed returns vs regular sales
- Built RFM scores for customer segmentation

## Python Libraries
- Pandas for data wrangling
- NumPy for calculations
- Matplotlib and Seaborn for charts

## About Me
**Ravi Teja**  
LinkedIn: linkedin.com/in/raviteja~  
Email: raviteja.inboxx@gmail.com  
GitHub: github.com/raviteja-k-01

I'm interested in data analytics and always looking to learn more. Feel free to reach out if you have questions or suggestions!

## License
MIT License - feel free to use this however you want

## Ideas for Future Work
Some things I'd like to add if I come back to this:
- Predict which customers might stop buying (churn prediction)
- Set up automatic data refresh
- Add more drill-down capabilities
- Make it mobile-friendly
- Analyze profit margins by product
- Send automated reports via email

## Contributing
If you spot any issues or have ideas for improvements, I'd love to hear them! Feel free to open an issue or submit a pull request.

---

Last updated: October 24, 2025
