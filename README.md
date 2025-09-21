# PFCP
Code for Nips 2025 paper Personalized Federated Conformal Prediction with Localization

The python version is 3.10.16 and library requirements are in requirements.txt

The Main folder contains the code for all methods, including GLCP, PFCP, FedCP, FedCP-QQ, CPlab, and CPhet, as well as methods that incorporate the agent screening process.

The Dataset folder contains all the data required for the six real datasets, where the dataset named "achieve" corresponds to STAR, while the remaining names are consistent with those used in the paper.

The experiment on synthetic data in Section 3.1 of the paper is located under the directory Test_CD, while the experiment on real data in Section 3.2 can be found under the six directories prefixed with Real.

The experiment in Appendix A4.1 is located in the folder Test_Aux, A4.2 in Test_AuxN, A4.3 in Test_Alp, and A4.4 to A4.5 in Sele_Aux.