# Glossary

Terms used in this project, in plain English. Written for readers new to housing finance,
statistics or machine learning.

Terms are grouped by topic, roughly in the order a reader meets them, not alphabetically.

---

## The thing being built

**AVM — Automated Valuation Model**
A computer model that estimates what a house is worth, without anyone visiting it. Banks
use them to decide how much to lend; websites use them for the price estimate on a
listing. This project builds one and then tests it the way a bank would.

**Hedonic model**
A model that prices something by pricing its *parts*. A house is valued as so much for the
square footage, so much for the garage and so much for the neighbourhood, added together. "Hedonic" just means "made of components".

**Valuation / appraisal**
An estimate of market value. An *appraisal* is done by a licensed human who visits the
property; a *valuation* from an AVM is a model's estimate. They are not legally
interchangeable, which is why the client report in notebook 10 says so explicitly.

**Arm's-length sale**
A normal sale between a willing buyer and a willing seller, neither under pressure. The
opposite is a foreclosure, a transfer between family members, or a half-built house sold
by the developer. None of these trade at market value. 17.6% of the sales in this data
are *not* arm's-length, which is why notebooks 03 and 08 spend so much effort on them.

---

## How accuracy is measured

**MdAPE — Median Absolute Percentage Error**
The headline accuracy number for an AVM. Take every valuation, work out how far off it
was as a percentage, and report the middle one. An MdAPE of 5% means a typical valuation
is out by about 5%. The median is used instead of the mean so that a single very large
miss cannot distort the score.

**PPE10 — Percentage of Predictions within 10%**
The share of valuations that landed within ±10% of what the house actually sold for.
"PPE10 of 76%" means three-quarters of the estimates were within a tenth of the truth.
There is also PPE5, PPE20 and PPE25: same idea, different tolerance. This is the number
a lending policy is usually written against.

**FSD — Forecast Standard Deviation**
A measure of how spread out a model's errors are. Vendors publish it as a confidence
figure. Lower is better.

**Bias / mean signed error**
Whether a model is wrong in a *consistent direction*. A model can be accurate on average
and still read 5% high on every cheap house. That is bias. It matters far more than random
error because it does not cancel out across a portfolio.

**RMSE — Root Mean Squared Error**
The average size of the error, in dollars, with big misses counted extra heavily. Common
in general statistics; less useful for valuation than MdAPE, because it is dominated by
the most expensive homes.

**R² ("R-squared")**
How much of the variation in price the model explains, from 0 to 1. A **negative** R²
means the model is worse than simply guessing the average price every time. This project's
first-pass submission had a negative R² once the true prices were recovered.

---

## Confidence and uncertainty

**Prediction interval / confidence range**
A range instead of a single number: "this house is worth $185,000, with 80% confidence
that the true value is between $168,000 and $205,000." More useful than a point estimate
because it shows how much to trust it.

**Conformal prediction**
The specific method used here to build those ranges. It makes almost no assumptions. Instead of
assuming errors follow a bell curve (they do not), it measures how wrong the model was on
past houses and uses that directly.

**Coverage**
The check on that promise. If a model says "80% confident" and the true price lands inside
the range 80% of the time, coverage is good. If it lands inside only 60% of the time, the
range is too narrow and the promise is false.

**Calibration**
Whether a stated probability is true. If a model says "70% likely" about a hundred things,
roughly seventy of them should happen. A model can rank things perfectly and still be
badly calibrated. Notebook 06 finds this problem and fixes it.

**Duan smearing**
A technical correction. The model works with the *logarithm* of price for statistical
reasons; converting back to dollars naively gives an answer that is systematically a bit
too low. Smearing corrects it. Small in size, but it biases every valuation in the same
direction, so it matters.

---

## Statistics you will meet

**Log transform**
Working with the logarithm of a number instead of the number. For prices this is natural,
because a $20,000 error on a $100,000 home and a $100,000 error on a $500,000 home are the
same *proportional* mistake. Logs put them on the same footing.

**Heteroskedasticity**
A long word for "the errors get bigger as the numbers get bigger". Predicting expensive
homes is harder in dollar terms than predicting cheap ones. It breaks several standard
statistical assumptions, and the log transform is the usual fix.

**Regression to the mean**
The tendency of predictions to sit closer to the average than the outcomes they predict.
When errors are grouped by the *true* value, any noisy model appears to over-value cheap
homes and under-value expensive ones, even one that is unbiased for the value a lender
actually sees. Notebook 04 shows that most of this project's price-decile "bias" is
this effect; the residual that remains is about 2–3% at the cheap end.

**Overfitting**
When a model memorises its training examples instead of learning the general pattern. It
looks excellent on data it has seen and fails on anything new.

**Leakage**
When information that would not be available in reality sneaks into training, for
instance letting the model see the answer. Scores on leaked data overstate how the model
will do on new data. Four of the ten defects in this project's first pass were leakage.

**Train / test split**
Holding some data back so the model can be graded on examples it has never seen.
*Random* split: shuffle and take a quarter. *Temporal* split: train on older sales, test
on newer ones. The temporal split is harder, and it is the only fair test for a model that
will be used forward in time.

**Cross-fitting**
A technique for using one model's output as another model's input without cheating. Split the
data into parts, and for each part use a model trained only on the *other* parts. Used in
notebook 06.

**Moran's I**
A test for whether errors cluster geographically. If a model gets houses on one side of
town wrong in the same direction, it has missed something about location. Around 0 means
no clustering; closer to 1 means strong clustering.

**PSI — Population Stability Index**
One number for "how much has this shifted?". Compares the data a model was built on
against the data it is seeing now. Below 0.10 means "same population"; above 0.25 means
"meaningfully different, go and look".

**Imbalanced classification**
A yes/no prediction problem where one answer is rare. Only 17.6% of sales are non-market,
so a model that says "market sale" every time is right 82.4% of the time and catches
none of them. Precision, recall and lift are used instead of accuracy.

**Precision and recall**
Of the cases flagged, the share that were real (**precision**); of the real cases, the
share that were caught (**recall**). Tightening a model's trigger raises precision and
lowers recall, and vice versa.

**Lift**
How much better than random a ranked list is. A lift of 4 in the top decile means
reviewing the riskiest 10% finds 40% of the problems.

---

## Property and lending terms

**LTV — Loan-to-Value**
The loan as a percentage of the property's value. An 80% LTV loan on a $300,000 house is a
$240,000 loan. The single most important number in mortgage credit risk, because it says
how much cushion the lender has if prices fall.

**MTM LTV — Mark-to-Market LTV**
The same ratio recalculated *today*: current loan balance against current property value.
It moves as the borrower pays down the loan and as house prices change.

**Negative equity / underwater**
When the loan is larger than the property is worth. Strongly associated with default,
because walking away starts to look rational.

**PD, LGD, EAD — the three pieces of expected loss**
- **PD** (Probability of Default): how likely the borrower is to stop paying.
- **LGD** (Loss Given Default): if they do stop, what share of the loan is actually lost
  after the property is repossessed and sold.
- **EAD** (Exposure at Default): how much is owed at that moment.

Multiply them together and you get **expected loss**, the amount a lender should expect
to lose on average. Usually quoted in **basis points**.

**Basis point (bp)**
One hundredth of a percent. 100 bps = 1%. Used because credit losses are small numbers
where "0.88%" is harder to compare than "88 bps".

**Stress test / CCAR / DFAST**
A regulatory exercise where banks calculate their losses under a deliberately terrible
scenario. The Federal Reserve publishes the scenario; the "severely adverse" one includes
house prices falling about 30%. Notebook 08 applies it.

**Cap rate — capitalisation rate**
The annual income a property produces as a percentage of its price. A $300,000 house
producing $18,000 a year after expenses has a 6% cap rate. The basic yardstick of property
investment.

**NOI — Net Operating Income**
Annual rent minus all the costs of running the property, but before the mortgage.
Excluding the mortgage is what makes it comparable between a cash buyer and a borrower.

**DSCR — Debt Service Coverage Ratio**
NOI divided by the annual mortgage payment. Below 1.0 the property cannot pay its own
mortgage. Lenders typically want 1.20–1.25.

**IRR — Internal Rate of Return**
The annualised return on an investment, accounting for when money goes in and comes out.

**Cash-on-cash return**
The first year's cash profit divided by the cash put in. Simpler than IRR, and ignores
everything after year one.

**Rollback (Iowa property tax)**
Iowa taxes only a fraction of a home's assessed value. That fraction is the "rollback",
set by the state each year: 47.4% for the 2024 assessment year. Combined with the local
levy it gives the effective tax rate used throughout this project: **1.4505%** of market
value.

---

## Governance and model risk

**SR 11-7**
The US banking regulators' rulebook for managing model risk, issued by the Federal Reserve
and the OCC. It says validating a model means three things: checking the design is sound,
checking the answers match reality, and **continuing to check after launch**. Notebook 05
exists because of that third one.

**Interagency AVM Rule (June 2024)**
A rule from six US financial regulators covering AVMs used in mortgage lending. It requires
firms to address five things: confidence in the estimates, protection against data
manipulation, conflicts of interest, random sample testing, and nondiscrimination.
Notebook 04 works through all five.

**Model validation**
Independent checking of a model by someone other than the person who built it. In this
project it is self-assessment, which is listed as a limitation.

**Champion / challenger**
The model currently in production (champion) versus a candidate replacement (challenger),
compared fairly before anyone switches. A challenger has to win *consistently*, not just
on average.

**Drift**
The world changing underneath a model, so the data it sees stops resembling the data it
learned from.

**Recalibration trigger**
A limit set *in advance* that, once crossed, forces the model to be rebuilt. A limit
set after seeing the results can simply be chosen to fit them.

---

## Data sources

**FHFA House Price Index**
A US government house price index built from repeat sales of the same homes. Used here for
the Ames market.

**Case-Shiller**
The best-known US national house price index. Used as the benchmark Ames is compared
against.

**PMMS**
Freddie Mac's weekly survey of average 30-year mortgage rates.

**ZHVI / ZORI**
Zillow's published indices for typical home *value* (ZHVI) and typical *rent* (ZORI) in a
metro area.

**CPI**
The Consumer Price Index, the standard measure of inflation. Used to convert past prices
into today's money.

**HMDA**
The Home Mortgage Disclosure Act. US lenders above a size threshold must publish, for
every mortgage application, what was asked for, what was decided, why it was refused, and
the applicant's race, ethnicity and sex. It exists so that lending outcomes can
be examined, and the data is free and public.

**Denial rate**
The share of applications that were turned down, out of those the lender actually decided.
Applications the borrower withdrew, and loans simply bought from another lender, are not
decisions and do not belong in the denominator.

**Odds ratio**
How much more likely one group is to see an outcome than another, after holding other
things equal. 1.0 means no difference; 2.0 means roughly twice the odds. Reported with a
range, because an odds ratio computed on 174 people is a much vaguer thing than one
computed on 20,000.

**Confounder**
Something you did not measure that could explain an apparent difference. If two groups are
denied at different rates and one group also has lower credit scores, credit score is a
confounder. Public mortgage data does not record it.

**E-value**
A number that answers "but what about the thing you did not measure?". It is how strong
that missing thing would have to be (how tightly tied to both the group and the outcome)
to explain the whole result away. A small E-value means the finding is fragile. A large
one means the missing variable would have to be very strong, and the reader can judge
whether that is plausible.

**Wilson interval**
A way of putting error bars on a percentage that behaves properly when the percentage is
small or the sample is tiny. The textbook method can produce a "range" that extends below
zero; this one cannot.

**Collateral denial**
A mortgage refused because the property did not support the loan (the valuation came in
too low), not because of anything about the borrower. It is what a valuation error
looks like from the applicant's side of the desk.
