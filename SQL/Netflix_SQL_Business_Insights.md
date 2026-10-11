# Netflix Customer Intelligence – SQL Business Insights

This file contains the business insights from the 37 SQL questions. SQL queries are maintained separately.

## Q1 – Active Customers by Country

- Active customers are distributed across multiple countries.
- Country-level customer volume can be used to identify major customer markets.

**Business Takeaway:** Country-level distribution can support regional marketing, support planning, and localization decisions.

## Q2 – Total and Average Monthly Revenue by Plan

- Standard has the highest total revenue: 35,255.24.
- Premium: 28,760.49; Basic: 17,349.18.
- Average monthly revenue per subscription is highest for Premium (22.99), followed by Standard (15.49) and Basic (6.99).

**Business Takeaway:** Standard is the largest total-revenue contributor, while Premium has the highest revenue per subscription.

## Q3 – Overall Churn Rate

- Overall churn rate is 24.89%.
- Approximately one in four customers is classified as churned.

**Business Takeaway:** Retention is an important area for further analysis.

## Q4 – Acquisition Channel and Churn

- Organic Search has the highest customer count (1,786).
- Paid Social has the highest reported churn rate (25.93%), followed closely by Direct (25.90%).
- Partner Bundle has the lowest reported churn rate (23.59%).

**Business Takeaway:** Acquisition channels should be evaluated using both acquisition volume and retention.

## Q5 – Average Age by Plan

- Basic: 32.68, Standard: 32.41, Premium: 32.47.
- Average age is very similar across plans.

**Business Takeaway:** Plan choice does not show a major age difference in this dataset.

## Q6 – Top 10 Most-Watched Titles

- Distant Harbor is the top title with 9,560 in the query's viewing measure.
- Other highly watched titles include Iron Kingdom Part 6, Fractured Frontier Part 2 and Wild Echo.

**Business Takeaway:** Popular titles can support content recommendation and investment analysis.

## Q7 – Average Completion by Content Type and Genre

- Completion is generally in the high-60% to low-70% range.
- Stand-Up Special / Sports is highest at 73.29%.
- Stand-Up Special / Stand-Up Comedy is lowest at 66.26%.

**Business Takeaway:** Completion varies by content category and can indicate stronger or weaker engagement.

## Q8 – Failed Payments

- The reported failed-payment result covers October 2025 through May 2026.
- Monthly counts were 98, 74, 84, 91, 66, 82, 96 and 88.

**Business Takeaway:** Failed-payment patterns can be monitored as a possible contributor to churn and support demand; this is not a full 12-month trend.

## Q9 – Average Resolution Time by Issue Category

- Streaming Quality has the highest average resolution time: 18.57 hours.
- Billing & Payments has the highest resolved/closed ticket volume (1,287) and the lowest average resolution time among categories: 17.30 hours.

**Business Takeaway:** Streaming Quality needs attention from a resolution-time perspective, while Billing & Payments represents the largest support workload.

## Q10 – Customer Satisfaction Distribution

- 3-star ratings are the largest group: 1,506 (32.80%).
- 4-star: 1,279 (27.85%); 5-star: 666 (14.50%).
- 1- and 2-star ratings together account for 24.84%.

**Business Takeaway:** Mid-level ratings form the largest group, while 4–5 star ratings exceed 1–2 star ratings.

## Q11 – Device with Most Total Viewing Minutes

- Use the final Q11 result from the SQL notebook here.

**Business Takeaway:** Device-level viewing patterns can guide platform support and viewing-experience optimization.

## Q12 – Average Feedback Rating by Sentiment

- Positive: 4.40 average rating.
- Neutral: 3.03.
- Negative: 1.64.

**Business Takeaway:** Sentiment aligns strongly with average customer ratings and can support customer-experience analysis.

## Q13 – Registrations by Acquisition Channel and Year

- Organic Search leads registrations among the acquisition channels.
- 2026 registration figures are incomplete.

**Business Takeaway:** Compare channels over time, but do not interpret incomplete 2026 data as a full-year result.

## Q14 – Auto-Renew Off

- 18.57% of customers have auto-renew turned off.

**Business Takeaway:** Customers without auto-renew can be monitored as a retention segment.

## Q15 – Customers with More Than 3 Support Tickets

- Multiple customers have more than three support tickets.
- Maximum observed ticket count is 7, for CUST007899 and CUST004986.

**Business Takeaway:** Repeated support interactions can identify customers with recurring service problems.

## Q16 – Plans Ranked by Total Revenue

- Standard ranks first: 35,255.24.
- Premium ranks second: 28,760.49.
- Basic ranks third: 17,349.18.

**Business Takeaway:** Standard is the largest total-revenue contributor.

## Q17 – Latest Support Ticket per Customer

- ROW_NUMBER() identifies the latest ticket for each customer.
- The output provides the latest issue, status and ticket date.

**Business Takeaway:** Latest-ticket information supports follow-ups and customer-health analysis.

## Q18 – Three-Month Moving Average of New Registrations

- Monthly registrations ranged from 71 to 111 in the reported result.
- Maximum monthly registrations: 111 in June 2021.
- Maximum 3-month moving average: 100.33 in November 2022.

**Business Takeaway:** A moving average smooths monthly fluctuations and highlights longer-term registration patterns.

## Q19 – Latest Monthly Watch Time vs Previous Month

- The query identifies months where watch time declined compared with the previous month.

**Business Takeaway:** Month-over-month watch-time declines can be used as an engagement signal and investigated alongside churn and content factors.

## Q20 – Cohort Retention

- Older cohorts have more complete follow-up periods.
- Recent cohorts have incomplete later checkpoints and should be interpreted cautiously.

**Business Takeaway:** Cohort analysis shows how retention changes through the customer lifecycle.

## Q21 – Top Countries by Churn Rate

- The raw result contained inconsistent country labels; normalization is required for reliable country rankings.
- Examples included S. Korea and Nigeria at 66.67% in the raw output.

**Business Takeaway:** Country-level churn comparisons should use normalized country values.

## Q22 – Running Total Payments per Customer

- A window function calculates cumulative payment totals for each customer over time.

**Business Takeaway:** Running spend supports lifetime-value analysis and customer segmentation.

## Q23 – Three Consecutive Months of Declining Watch Time

- The query identifies customers/months with three consecutive months of declining watch time.

**Business Takeaway:** Sustained engagement decline can act as an early-warning signal for churn analysis.

## Q24 – Monthly Churn Rate

- Churn increased from 2.72% in October 2025 to 4.01% in June 2026.
- Highest reported value: 4.01% in June 2026.

**Business Takeaway:** The reported increase warrants investigation into engagement, payments, pricing and support factors.

## Q25 – Retention by Acquisition Channel

- Partner Bundle has the highest reported retention: 76.41%.
- Paid Social has the lowest reported retention: 74.07%.

**Business Takeaway:** Acquisition channels can be evaluated by both volume and subsequent retention.

## Q26 – Lifetime Spend Quartiles

- NTILE(4) divides customers into four lifetime-spend groups.
- The top quartile contains customers with the greatest cumulative payment value.

**Business Takeaway:** Spend quartiles support customer segmentation and can be combined with churn.

## Q27 – Top 3 Customer Complaints per Country

- India: Invoice question (63), Failed payment (58), Refund request (56).
- United States: Refund request (78), Buffering (73), Audio sync issue (67).
- United Kingdom: Buffering (32), Duplicate charge (31), Failed payment (30).
- Germany: Audio sync issue (25), Cancellation request (22), Refund request (22).

**Business Takeaway:** Complaint patterns differ by country, supporting country-specific support priorities.

## Q28 – Customers with 2+ Failed Payments in Latest 90 Days

- The query identifies active customers with at least two failed payments in the latest 90 days.
- Examples include CUST000662, CUST002514, CUST005057, CUST000439 and CUST004108.

**Business Takeaway:** Repeated payment failures can be monitored as a potential risk signal, but the query has no formal high-risk label.

## Q29 – Engagement Before/After Subscription Price Change

- No subscription price-change events were available in the dataset.

**Business Takeaway:** A before/after price-change analysis could not be performed with the available data.

## Q30 – Country Customer, Churn, Churn Rate and LTV

- Raw country output contained inconsistent labels before normalization.
- Example results: India 1,065 customers, 275 churned, 25.82% churn; United States 1,253 customers, 312 churned, 24.90% churn.

**Business Takeaway:** Country-level reporting should use normalized country values.

## Q31 – At-Risk Revenue Report

- The query identifies active customers using a project-defined risk score and associated revenue.
- Example high-risk-score records included CUST005811 and CUST005563.

**Business Takeaway:** Combining revenue with a project-defined risk score helps prioritize retention monitoring; the score is a project heuristic.

## Q32 – Top 5 Customer Complaints per Country

- India: Low resolution (9), Refund request (8), Account locked (6), Cancellation request (6), Failed payment (6).
- United States: Audio sync issue (8), Buffering (7), Download failure (7), Login failure (7), Failed payment (6).
- United Kingdom: Low resolution (4), Buffering (3), Cancellation request (3), Duplicate charge (3), Password reset (3).
- Brazil: App crashing (4), Duplicate charge (3), Missing subtitles (3), Password reset (3), Refund request (3).

**Business Takeaway:** Complaint patterns vary by country, supporting localized support and product improvements.

## Q33 – Win-Back Candidates

- The query identifies churned customers who are in the top lifetime-value quartile.

**Business Takeaway:** This segment combines previous churn with high historical value and can be used for targeted win-back analysis.

## Q34 – Plan Migration Analysis

- Highest plan-pair counts include Basic→Basic (128), Standard→Basic (119), Premium→Basic (114), and Standard→Standard (108).
- Same-plan pairs are not actual migrations; different-plan pairs represent plan changes.

**Business Takeaway:** Plan transitions help understand upgrade and downgrade behavior.

## Q35 – WAU, MAU and Stickiness

- April 2026: average weekly active users 1,005.50; MAU 3,191; stickiness 0.3151.
- May 2026: 1,622.75; MAU 4,302; stickiness 0.3772.
- June 2026: 1,426.80; MAU 4,235; stickiness 0.3369.

**Business Takeaway:** WAU/MAU shows how frequently monthly users return on a weekly basis.

## Q36 – Negative Feedback After Failed Payment or Low-Satisfaction Ticket

- The query identified 41 matching records.
- CUST003181 appears twice.

**Business Takeaway:** Negative feedback near payment failures or low-satisfaction support interactions can reveal customer-experience patterns.

## Q37 – Customer Health Score

- The health score combines multiple customer signals using a project-defined scoring approach.
- Some records have missing current-plan information.

**Business Takeaway:** A combined health score can provide a monitoring metric for customer engagement and risk, but its formula should be documented as project-specific.

## Overall SQL Project Summary

The SQL analysis covers customer and revenue analysis, churn and retention, acquisition channels, content engagement, payments, customer support, feedback, lifetime value, plan migration, and customer health/risk signals.

These insights form the foundation for the next stages: EDA, churn prediction, feature engineering, model evaluation, and the AI Analytics Copilot.