# loan_default_pytorch
A deep-learning predictive model to define if a loan will default. It is based on PyTorch and uses the Kaggle competition data "SBA Loans Case Data Set". The model provides the framework that after being retrained with the updated and related data, can advise banks and financial institutions in the approval process of their loan applicants.


# The Kaggle story and dataset:

SBA Loans Case Data Set

About Dataset
Should This Loan be Approved or Denied?
If you like the data set and download it, an upvote would be appreciated.

The Small Business Administration (SBA) was founded in 1953 to assist small businesses in obtaining loans. Small businesses have been the primary source of employment in the United States. Helping small businesses help with job creation, which reduces unemployment. Small business growth also promotes economic growth. One of the ways the SBA helps small businesses is by guaranteeing bank loans. This guarantee reduces the risk to banks and encourages them to lend to small businesses. If the loan defaults, the SBA covers the amount guaranteed, and the bank suffers a loss for the remaining balance.

There have been several small business success stories like FedEx and Apple. However, the rate of default is very high. Many economists believe the banking market works better without the assistance of the SBA. Supporter claim that the social benefits and job creation outweigh any financial costs to the government in defaulted loans.

The Data Set
The original data set is from the U.S.SBA loan database, which includes historical data from 1987 through 2014 (899,164 observations) with 27 variables. The data set includes information on whether the loan was paid off in full or if the SMA had to charge off any amount and how much that amount was. The data set used is a subset of the original set. It contains loans about the Real Estate and Rental and Leasing industry in California. This file has 2,102 observations and 35 variables. The column Default is an integer of 1 or zero, and I had to change this column to a factor.

For more information on this data set go to https://amstat.tandfonline.com/doi/full/10.1080/10691898.2018.1434342

Variable descriptions:

LoanNr_ChkDgt:         Identifier – Primary key
NAICS:                 North American industry classification system code
ApprovalDate:          Date SBA commitment issued
ApprovalFY:            Fiscal year of commitment
Term:                  Loan term in months
NoEmp:                 Number of Business Employees
NewExist:              1 = Existing business, 2 = New business
CreateJob:             Number of Jobs Created
RetainedJob:           Number of jobs retained
FranchiseCode:         Franchise code, (00000 or 00001) = No franchise
UrbanRural:            1 = Urban, 2 = rural, 0 = undefined
RevLineCr:             Revolving line of credit: Y = Yes, N = No
LowDoc:                LowDoc Loan Program: Y = Yes, N = No
ChgOffDate:            The date when a loan is declared to be in default
DisbursementDate:      Disbursement date
DisbursementGross:     Amount disbursed
BalanceGross:          Gross amount outstanding
MIS_Status:            Loan status charged off = CHGOFF, Paid in full = PIF 
ChgOffPrinGr:          Charged-off amoun
GrAppv:                Gross amount of loan approved by bank
SBA_Appv:              SBA's guaranteed amount of approved loan
New:                   New or Existing Loan
RealEstate:            Was Real Estate Used as Collateral
Portion:               What Portion of the Loan was Guaranteed by the SBA
Recession:             Was this loan made during a Recession
daysterm:              How many Days were in the Loan Terms
xx:                    Amount of Default if Any
Default:               Did the Loan Default

Explanation: A "charge-off" in finance and accounting refers to a creditor or lender writing off an unpaid debt as a loss, essentially saying they don't expect to be paid back.


Files:

The dev folder includes the development notebook with all the EDA and model development stage.
The dataset is in the data folder.
The production code is in src folder.




