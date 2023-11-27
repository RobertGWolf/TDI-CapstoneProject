


# TDI-CapstoneProject:  Predicting Heat Treatment from Micrographs

Welcome to my TDI capstone project where I build a TensorFlow model that can recover the heat treatment of steel from micrographs.

What is a micrograph? It is a picture of the microstructure of the steel, the microscopic architecture. This microstructure is formed via the heat treatment of the steel and determines its physcial proeprties.

<p align="center">
<img src="https://github.com/RobertGWolf/TDI-CapstoneProject/assets/133603510/887dcb03-27fa-41ef-967a-5f9e1db4ac60" />
<p align = "center">Fig 1.  Microsgraph of Ultra High Carbon Steel heat treated at 970 °C for 8 Hours</p>
</p>.

## The Dataset

The dataset for this project is the Ultra High Carbon Steel dataset archived by <a href = "materialsdata.nist.gov">NIST</a>.  It containst approximately 600 micrographs of annealed UHCS samples with anneal times between 5 an 5100 minutes and anneal temepratures between 700 °C and 1100 °C.

## The Model
### Data Augmentation: Images
  Due to the low number of images, data augmentation of the image set is needed.  
  *example picture of of augmented data, use micrograph 10*
### Data Transformation: Time and Temperature, analysis of Diffusion Equation
$\phi(x,t) = \frac{1}{\sqrt{4\pi Dt}}e^{-\frac{x^2}{4Dt}}$

$D = D_0 e^{-\frac{E_A}{RT}}$
### Data Transformation: Cooling Methods, one hot encoding
### Transfer Learning: InceptionV3
### Deep Learning Feed Forward Network
## Model Results:
### Metrics: R-squared, Mean Absolute Error, Difference Scatter Plot
### Metrics for naive Time, Temperature Data: negative R-squared values.
