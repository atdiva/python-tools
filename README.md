Sample files:

* datascience

    * RealTimeSurgeDetection.py  
    **Question: When do we expect the peak surge time to occur under uncertainty?**  
    Description: Implements an Extended Kalman Filter to track and estimate the timing and amplitude of transient surge events in noisy signals in real time.

  * MahalanobisOutlierDetection.py  
    **Question: Which observations in a dataset are statistically inconsistent with the underlying distribution?**  
    Description: Detects multivariate outliers using Mahalanobis distance and covariance structure.

  * RegimeSwitchDetect.py  
    **Question: Where do statistically significant changes in distribution occur within a time series?**  
    Description: Detects regime shifts by comparing distributions before and after candidate change points.

* geospatial
  * ChicagoGrocery.py  
    **Question: How far is each county’s population center from the nearest grocery store, and where do potential food access gaps exist?**  
    Description: Computes geodesic distance from county centroids to nearest grocery locations to quantify spatial accessibility.

  * CobaltLikelihood.py  
    **Question: How does proximity to key geophysical features influence the likelihood of cobalt mineralization?**  
    Description: Estimates spatial likelihood of cobalt presence using distance-based decay from relevant geological structures or contacts.

  * MLGeologyClassification.py  
    **Question: Can geochemical or spatial features reliably distinguish between sedimentary and metamorphic rock samples?**  
    Description: Trains a logistic regression classifier to predict rock type and evaluate feature separability.

  * TornadoDamageRegression.py  
    **Question: How well do measurable tornado characteristics predict downstream impacts like damage, injuries, or fatalities?**  
    Description: Uses regression modeling to quantify relationships between tornado features and observed outcome severity.

* orbital
  * 3I-ATLAS.py  
    **Question: Given current state estimates, what trajectory does 3I-ATLAS follow under heliocentric two-body dynamics?**  
    Description: Propagates and visualizes the object’s orbit using classical Keplerian motion assumptions.

  * KeplerianOrbit.py  
    **Question: How can we efficiently propagate orbital states under two-body dynamics?**  
    Description: Implements a reusable class for Keplerian orbit propagation and state vector evolution.

  * OrbitalIntercept.py  
    **Question: What is the minimum-effort trajectory required for an object to intercept Earth under two-body dynamics?**  
    Description: Computes an optimal intercept trajectory using simplified heliocentric dynamics and control assumptions.

* sim
  * Traffic.py  
    **Question: How do simple acceleration and braking rules (bang-bang control) produce emergent traffic flow behavior?**  
    Description: Simulates 1D traffic dynamics to explore congestion formation and stability under discrete control laws.

  * CompetitiveSpecies.py  
    **Question: How do interacting species populations evolve over time under nonlinear competition dynamics?**  
    Description: Numerically solves coupled Lotka–Volterra equations to analyze stability, oscillations, and equilibrium behavior.



    
