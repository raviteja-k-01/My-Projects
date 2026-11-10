-- Phase-1: Uber Ride Analytics — robust load, cleaning, and exploration queries
-- Adjust file path and LINES TERMINATED BY if needed ('\n' or '\r\n').

-- 0. Create and use database
CREATE DATABASE IF NOT EXISTS uber_ride_analytics;
USE uber_ride_analytics;

-- 1. Create table (drop if exists to ensure repeatable runs)
DROP TABLE IF EXISTS bookings;
CREATE TABLE bookings (
    Date DATE,
    Time TIME,
    Booking_ID VARCHAR(128),
    Booking_Status VARCHAR(128),
    Customer_ID VARCHAR(128),
    Vehicle_Type VARCHAR(64),
    Pickup_Location VARCHAR(255),
    Drop_Location VARCHAR(255),
    Avg_VTAT DECIMAL(8,2),    -- pickup wait (minutes)
    Avg_CTAT DECIMAL(8,2),    -- trip time (minutes)
    Cancelled_Rides_by_Customer INT,
    Reason_for_cancelling_by_Customer VARCHAR(400),
    Cancelled_Rides_by_Driver INT,
    Driver_Cancellation_Reason VARCHAR(400),
    Incomplete_Rides INT,
    Incomplete_Rides_Reason VARCHAR(400),
    Booking_Value DECIMAL(14,2),
    Ride_Distance DECIMAL(10,2),
    Driver_Ratings DECIMAL(3,1),
    Customer_Rating DECIMAL(3,1),
    Payment_Method VARCHAR(128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 2. Robust LOAD DATA: strip currency symbols and thousands separators, map ''/'null' -> NULL
-- Replace LINES TERMINATED BY '\n' with '\r\n' if your CSV uses CRLF.
LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\ncr_ride_bookings.csv'
INTO TABLE bookings
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(@Date, @Time, @Booking_ID, @Booking_Status, @Customer_ID, @Vehicle_Type, @Pickup_Location, @Drop_Location,
 @Avg_VTAT, @Avg_CTAT, @Cancelled_Rides_by_Customer, @Reason_for_cancelling_by_Customer,
 @Cancelled_Rides_by_Driver, @Driver_CANCELLATION_REASON, @Incomplete_Rides, @Incomplete_Rides_Reason,
 @Booking_Value, @Ride_Distance, @Driver_Ratings, @Customer_Rating, @Payment_Method)
SET
  Date = STR_TO_DATE(NULLIF(TRIM(@Date), ''), '%Y-%m-%d'),
  Time = NULLIF(TRIM(@Time), ''),
  Booking_ID = NULLIF(TRIM(@Booking_ID), ''),
  Booking_Status = NULLIF(TRIM(@Booking_Status), ''),
  Customer_ID = NULLIF(TRIM(@Customer_ID), ''),
  Vehicle_Type = NULLIF(TRIM(@Vehicle_Type), ''),
  Pickup_Location = NULLIF(TRIM(@Pickup_Location), ''),
  Drop_Location = NULLIF(TRIM(@Drop_Location), ''),
  -- numeric/text fields: convert literal 'null' and '' to SQL NULL before casting
  Avg_VTAT = NULLIF(NULLIF(TRIM(@Avg_VTAT), ''), 'null'),
  Avg_CTAT = NULLIF(NULLIF(TRIM(@Avg_CTAT), ''), 'null'),
  Cancelled_Rides_by_Customer = NULLIF(NULLIF(TRIM(@Cancelled_Rides_by_Customer), ''), 'null'),
  Cancelled_Rides_by_Driver = NULLIF(NULLIF(TRIM(@Cancelled_Rides_by_Driver), ''), 'null'),
  Incomplete_Rides = NULLIF(NULLIF(TRIM(@Incomplete_Rides), ''), 'null'),
  -- Booking_Value: strip commas/currency tokens, convert 'null' -> NULL, then empty -> NULL
  Booking_Value = NULLIF(
      NULLIF(
        TRIM(
          REPLACE(
            REPLACE(
              REPLACE(
                REPLACE(
                  REPLACE(@Booking_Value, ',', ''), '₹', ''), 'Rs.', ''), 'INR', ''), '$', ''
            )
        ), 'null'
      ), ''
  ),
  -- Ride_Distance: strip commas and map 'null' -> NULL
  Ride_Distance = NULLIF(NULLIF(TRIM(REPLACE(@Ride_Distance, ',', '')), 'null'), ''),
  -- Ratings: map 'null' and '' to NULL
  Driver_Ratings = NULLIF(NULLIF(TRIM(@Driver_Ratings), ''), 'null'),
  Customer_Rating = NULLIF(NULLIF(TRIM(@Customer_Rating), ''), 'null'),
  -- remaining text field
  Payment_Method = NULLIF(TRIM(@Payment_Method), '');

SELECT
  COUNT(*) AS total_rows,
  COUNT(Booking_Value) AS non_null_booking_values,
  (COUNT(*) - COUNT(Booking_Value)) AS missing_booking_value
FROM bookings;

SELECT
  COUNT(*) AS completed_count,
  COUNT(Booking_Value) AS completed_with_value,
  ROUND(AVG(Booking_Value),2) AS avg_booking_value_completed
FROM bookings
WHERE Booking_Status = 'completed';

SELECT
  SUM(CASE WHEN Booking_Status = 'cancelled by driver' THEN 1 ELSE 0 END) AS cancelled_by_driver_count,
  ROUND( (SUM(CASE WHEN Booking_Status = 'cancelled by driver' THEN 1 ELSE 0 END)
         * (SELECT AVG(Booking_Value) FROM bookings WHERE Booking_Status = 'completed' AND Booking_Value IS NOT NULL)
        ), 2) AS estimated_uncaptured_revenue
FROM bookings;

SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN Booking_Status = 'cancelled by driver' THEN 1 ELSE 0 END) AS cancelled_by_driver,
  ROUND(100.0 * SUM(CASE WHEN Booking_Status = 'cancelled by driver' THEN 1 ELSE 0 END) / COUNT(*),2) AS pct_cancelled_by_driver
FROM bookings;

SELECT
  COALESCE(HOUR(Time), -1) AS ride_hour,
  COUNT(*) AS total_rides,
  SUM(CASE WHEN Booking_Status = 'cancelled by driver' THEN 1 ELSE 0 END) AS driver_cancels,
  ROUND(100.0 * SUM(CASE WHEN Booking_Status = 'cancelled by driver' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS cancellation_rate_pct
FROM bookings
GROUP BY ride_hour
ORDER BY ride_hour;

SELECT
  Pickup_Location,
  COUNT(*) AS total_rides,
  SUM(CASE WHEN Booking_Status = 'cancelled by driver' THEN 1 ELSE 0 END) AS driver_cancels,
  ROUND(100.0 * SUM(CASE WHEN Booking_Status = 'cancelled by driver' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS cancellation_rate_pct
FROM bookings
GROUP BY Pickup_Location
HAVING total_rides >= 50
ORDER BY cancellation_rate_pct DESC
LIMIT 20;

SELECT
  Booking_Status,
  COUNT(*) AS ro_s,
  COUNT(Booking_Value) AS rows_with_value,
  ROUND(AVG(Booking_Value),2) AS avg_booking_value
FROM bookings
GROUP BY Booking_Status
ORDER BY ro_s DESC;

SELECT
  COUNT(*) AS driver_cancel_total,
  SUM(CASE WHEN Avg_VTAT IS NULL THEN 1 ELSE 0 END) AS driver_cancel_missing_vtat
FROM bookings
WHERE Booking_Status = 'cancelled by driver';


