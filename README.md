


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
The microstructure of the steel is a result diffusion, so it is sensible to consult the diffusion equation when decided how best the time and temperature data enter into the model.  The solution to the one dimensional equation is given by $\phi$ with diffusivity $D$.

$$\phi(x,t) = \frac{1}{\sqrt{4\pi Dt}}e^{-\frac{x^2}{4Dt}}$$

$$D = D_0 e^{-\frac{E_A}{RT}}$$

The solution is non-lienar with a singularity at $t=0$.  However if we neglect the singularity and apply a natural log we get the following.

$$\ln{e^{-\frac{x^2}{4Dt}}} = -\frac{x^2}{4Dt}$$

Applying a second natuarl logs gives a formula that is linear in inverse temperature and log time.  This is the movitvation for transforming the time/temperature data.

$$ \begin{align*}
\ln{\frac{4Dt}{x^2}}&= \ln{D} + \ln{t} + \ln{\frac{4}{x^2}}\\
&= \ln{D_0 e^{-\frac{E_A}{RT}}} + \ln{t} + \ln{\frac{4}{x^2}}\\
& = -\frac{E_A}{R}\frac{1}{T} + \ln{t} +\ln{\frac{4D_0}{x^2}}
\end{align*}$$

This motivation is vindicated as using untransformed data reusults in a model with negative R-squared values.  Running a smaller model with untransformed data results in R-squared values of $-0.7$ for temperature and $-0.6$ for time after 20 epochs.

### Data Transformation: Cooling Methods and One Hot Encoding
Cooling data is included in the metadata for each sample.  Diffusion in steel is driven time and temperature so it is reasonable to assume that the cooling method effects the microstructure.  Two samples annealed at 900 °C for 180 minutes may have significant differences in their microstructures if one is furnace cooled and the other is air cooled.  To account for this in the model the cooling methods were one hot encoded for the regressor to predict.

### Transfer Learning: InceptionV3
Transfer learning is a central tool of the model.  InceptionV3 with the top layer removed is used for image processing with inputs scaled to the cropped pictures.  The layers are initially set to untrainable until sufficient converegence for the model.  The training layers are then turned back on to fine tune the model.  

### Deep Learning Feed Forward Network

Two hidden layers are used as there is sufficient non-linearity in the problem even with the data transformation.  Batch noarmalization and dropout were the primary forms of regularization for the model which showed robust resistance to overfitting.  

## Model Results:
### Metrics: R-squared, Mean Absolute Error, Difference Scatter Plot

The model uses Mean Squared Error for optimization, however the two primary metrics for the model are R-squared and Mean Absolute Error.
chart: MAE for time and temp (14.8 °C  and 251 minutes) 
chart: R-squared (.91 for T and .85 for time)
note that there is a lot more variance in the time

we can also get a feel for the accuracy from the bullseye graph
show graph for all data
show graph for cribbed data



### Metrics for naive Time, Temperature Data: negative R-squared values.
