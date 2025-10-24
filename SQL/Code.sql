-- step 1: creating a database and importing a cleaned csv as a table in this database.
create database retail;
use retail;
select count(*) from cleaned;
SELECT * FROM cleaned LIMIT 10;
-- Step 2: Core Business SQL Queries
-- i)total sales and transactions
SELECT 
    COUNT(DISTINCT InvoiceNo) AS Total_Transactions,
    ROUND(SUM(Revenue), 2) AS Total_Sales
FROM cleaned;
-- ii)top 10 countries by sales
SELECT Country, ROUND(SUM(Revenue), 2) AS Total_Sales
FROM cleaned
GROUP BY Country
ORDER BY Total_Sales DESC
LIMIT 10;
-- iii)Monthly Sales Trend
SELECT Year, Month, ROUND(SUM(Revenue), 2) AS Monthly_Revenue
FROM cleaned
GROUP BY Year, Month
ORDER BY Year, Month;
-- iv) best selling products
SELECT Description, SUM(Quantity) AS Units_Sold, ROUND(SUM(Revenue), 2) AS Total_Revenue
FROM cleaned
GROUP BY Description
ORDER BY Total_Revenue DESC
LIMIT 10;
-- v)top 10 customers by spending
SELECT CustomerID, ROUND(SUM(Revenue), 2) AS Total_Spent, COUNT(DISTINCT InvoiceNo) AS Purchases
FROM cleaned
GROUP BY CustomerID
ORDER BY Total_Spent DESC
LIMIT 10;
-- vi)Weekday sales Distribution
SELECT DayOfWeek, ROUND(SUM(Revenue), 2) AS Total_Revenue
FROM cleaned
GROUP BY DayOfWeek
ORDER BY FIELD(DayOfWeek, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday');
-- vii)Quater =wise Performance 
SELECT Year, Quarter, ROUND(SUM(Revenue), 2) AS Revenue
FROM cleaned
GROUP BY Year, Quarter
ORDER BY Year, Quarter;
-- viii)Return Analysis
SELECT TransactionType, COUNT(*) AS Count, ROUND(SUM(Revenue), 2) AS Total_Revenue
FROM cleaned
GROUP BY TransactionType;

-- Step 3: SQL Business Analysis in MySQL
-- 1)Basic Data Verification
SELECT COUNT(*) AS Total_Rows, COUNT(DISTINCT InvoiceNo) AS Unique_Invoices, COUNT(DISTINCT CustomerID) AS Unique_Customers
FROM cleaned;
-- Total revenue Generated
SELECT ROUND(SUM(Revenue), 2) AS Total_Revenue
FROM cleaned;
-- top 10 best performing countries
SELECT Country, ROUND(SUM(Revenue), 2) AS Total_Sales
FROM cleaned
GROUP BY Country
ORDER BY Total_Sales DESC
LIMIT 10;
-- Monthly Revenue trend
SELECT Year, Month, ROUND(SUM(Revenue), 2) AS Monthly_Revenue
FROM cleaned
GROUP BY Year, Month
ORDER BY Year, Month;
-- Highest-Selling Products 
SELECT Description, SUM(Quantity) AS Units_Sold, ROUND(SUM(Revenue), 2) AS Total_Revenue
FROM cleaned
GROUP BY Description
ORDER BY Total_Revenue DESC
LIMIT 10;
-- top 10 customers spending 
SELECT CustomerID, ROUND(SUM(Revenue), 2) AS Total_Spent, COUNT(DISTINCT InvoiceNo) AS Purchase_Count
FROM cleaned
GROUP BY CustomerID
ORDER BY Total_Spent DESC
LIMIT 10;
-- Sales by day of week
SELECT DayOfWeek, ROUND(SUM(Revenue), 2) AS Total_Revenue
FROM cleaned
GROUP BY DayOfWeek
ORDER BY FIELD(DayOfWeek, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday');
-- Quater-Wise Perfofrmance
SELECT Year, Quarter, ROUND(SUM(Revenue), 2) AS Revenue
FROM cleaned
GROUP BY Year, Quarter
ORDER BY Year, Quarter;
-- Return vs Sales
SELECT TransactionType, COUNT(*) AS Transactions, ROUND(SUM(Revenue), 2) AS Total_Revenue
FROM cleaned
GROUP BY TransactionType;
-- Country with Highest Returns 
SELECT Country, COUNT(*) AS Returns, ROUND(SUM(Revenue), 2) AS Return_Value
FROM cleaned
WHERE TransactionType = 'Return'
GROUP BY Country
ORDER BY Returns DESC
LIMIT 5;
-- Product Category Analysis 
SELECT Category, ROUND(SUM(Revenue), 2) AS Total_Revenue, SUM(Quantity) AS Total_Quantity
FROM cleaned
GROUP BY Category
ORDER BY Total_Revenue DESC
LIMIT 10;

-- Step-4: RFM Analysis

-- Check the maximum  date in dataset 
select max(invoicedate) as Last_Date from cleaned;
-- Compute core RFM metrics
SELECT
    CustomerID,
    ROUND(SUM(Revenue), 2) AS Monetary,
    COUNT(DISTINCT InvoiceNo) AS Frequency,
    DATEDIFF((SELECT MAX(InvoiceDate) FROM cleaned), MAX(InvoiceDate)) AS Recency
FROM cleaned
GROUP BY CustomerID;
-- Assign RFM scores (1-5 Scale)
SELECT
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    NTILE(5) OVER (ORDER BY Recency DESC) AS R_Score,
    NTILE(5) OVER (ORDER BY Frequency ASC) AS F_Score,
    NTILE(5) OVER (ORDER BY Monetary ASC) AS M_Score
FROM (
    SELECT
        CustomerID,
        DATEDIFF((SELECT MAX(InvoiceDate) FROM cleaned), MAX(InvoiceDate)) AS Recency,
        COUNT(DISTINCT InvoiceNo) AS Frequency,
        ROUND(SUM(Revenue), 2) AS Monetary
    FROM cleaned
    GROUP BY CustomerID
) rfm_base;
-- Combine Score and Segment customers 
SELECT
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    CONCAT(
        NTILE(5) OVER (ORDER BY Recency DESC),
        NTILE(5) OVER (ORDER BY Frequency ASC),
        NTILE(5) OVER (ORDER BY Monetary ASC)
    ) AS RFM_Score
FROM (
    SELECT
        CustomerID,
        DATEDIFF((SELECT MAX(InvoiceDate) FROM cleaned), MAX(InvoiceDate)) AS Recency,
        COUNT(DISTINCT InvoiceNo) AS Frequency,
        ROUND(SUM(Revenue), 2) AS Monetary
    FROM cleaned
    GROUP BY CustomerID
) rfm_base;

-- Optional segmentaion example
SELECT
    CustomerID,
    CASE
        WHEN RFM_Score >= '444' THEN 'Loyal Customers'
        WHEN RFM_Score LIKE '5__' THEN 'At Risk'
        WHEN RFM_Score LIKE '_1_' THEN 'New Customers'
        WHEN RFM_Score LIKE '__5' THEN 'Big Spenders'
        ELSE 'Regulars'
    END AS Customer_Segment
FROM (
    SELECT
        CustomerID,
        CONCAT(
            NTILE(5) OVER (ORDER BY Recency DESC),
            NTILE(5) OVER (ORDER BY Frequency ASC),
            NTILE(5) OVER (ORDER BY Monetary ASC)
        ) AS RFM_Score
    FROM (
        SELECT
            CustomerID,
            DATEDIFF((SELECT MAX(InvoiceDate) FROM cleaned), MAX(InvoiceDate)) AS Recency,
            COUNT(DISTINCT InvoiceNo) AS Frequency,
            ROUND(SUM(Revenue), 2) AS Monetary
        FROM cleaned
        GROUP BY CustomerID
    ) base
) seg;

-- step-5: KPI Queries & Dashboard Export
-- i)Key Queries & Dashboard Export 
-- Toral Sales and orders 
SELECT 
    COUNT(DISTINCT InvoiceNo) AS Total_Orders,
    ROUND(SUM(Revenue), 2) AS Total_Sales,
    COUNT(DISTINCT CustomerID) AS Unique_Customers
FROM cleaned;
-- Average Order Value(AOV)
SELECT 
    ROUND(SUM(Revenue)/COUNT(DISTINCT InvoiceNo), 2) AS Average_Order_Value
FROM cleaned;
-- customer retention rate
SELECT 
    Month, 
    COUNT(DISTINCT CustomerID) AS Customers,
    COUNT(DISTINCT CASE WHEN PurchaseCount > 1 THEN CustomerID END) AS Repeat_Customers
FROM (
    SELECT CustomerID, Month, COUNT(InvoiceNo) AS PurchaseCount
    FROM cleaned
    GROUP BY CustomerID, Month
) monthly_purchases
GROUP BY Month
ORDER BY Month;
-- Top Product Categories by Revenue & Quantity
SELECT Category, 
       ROUND(SUM(Revenue), 2) AS Total_Revenue, 
       SUM(Quantity) AS Total_Quantity
FROM cleaned
GROUP BY Category
ORDER BY Total_Revenue DESC
LIMIT 10;