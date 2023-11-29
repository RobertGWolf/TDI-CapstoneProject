


# TDI-CapstoneProject:  Predicting Heat Treatment from Micrographs

Welcome to my TDI capstone project where I build a TensorFlow model that can recover the heat treatment of steel from a micrograph.

What is a micrograph? It is a picture of the microstructure of the steel, it's microscopic architecture. This microstructure is formed via the heat treatment of the steel and determines it's physcial proeprties.

<p align="center">
<img src="https://github.com/RobertGWolf/TDI-CapstoneProject/assets/133603510/887dcb03-27fa-41ef-967a-5f9e1db4ac60" />
<p align = "center">Fig 1.  Micrograph of Ultra High Carbon Steel heat treated at 970 °C for 8 Hours</p>
</p>.

## The Dataset

The dataset for this project is the Ultra High Carbon Steel dataset archived by <a href = "https://materialsdata.nist.gov">NIST</a>.  It contains approximately 600 micrographs of annealed UHCS samples with anneal times between 5 an 5100 minutes and anneal temepratures between 700 °C and 1100 °C.

## The Model
The model architecture consists of three main pieces: data augmentation and preproceing, tranfer learning using the inceptionV3 model, and a feed-forward deep neural network.

<p align="center">
<img src="https://github.com/RobertGWolf/TDI-CapstoneProject/assets/133603510/6d0fdde4-38ca-44f5-9663-55dc272e5f70" />
<p align = "center">Fig 2.  Augemnted Varation of Fig 1.</p>
</p>


### Data Augmentation: Images
Due to the low number of images, data augmentation of the image set is needed. An image is randomly flipped, rotated, translated, scaled and its contrast altered before being processed.    

<p align="center">
<img src="https://github.com/RobertGWolf/TDI-CapstoneProject/assets/133603510/c6815f0a-e435-4a61-8b1b-f2ec53bf3293" />
<p align = "center">Fig 3.  Model Architecture.</p>
</p>.

### Data Transformation: Time and Temperature, analysis of Diffusion Equation
The temperature is converted from Celsius to Kelvin to validate scaling, the inverse is taken and then normalized by the standard scaler.  Time is given in hours and minutes in the metadata and is converted to minutes before taking the log an normalizing by the standard scaler.

Celsius to Kelvin, hours to minutes, and standard sclaing are straight forward conversions.  Justifying using inverse temperature and log time requires an examination of the underlying physics.  The microstructure of the steel is a result diffusion, so it is sensible to consult the diffusion equation when decided how best the time and temperature data enter into the model.  The solution to the one dimensional equation is given by $\phi$ with diffusivity $D$.

$$\phi(x,t) = \frac{1}{\sqrt{4\pi Dt}}e^{-\frac{x^2}{4Dt}}$$

$$D = D_0 e^{-\frac{E_A}{RT}}$$

The solution is non-linear with a singularity at $t=0$.  However if we neglect the singularity and apply a natural log we get the following.

$$\ln{e^{-\frac{x^2}{4Dt}}} = -\frac{x^2}{4Dt}$$

Applying a second natural log gives a formula that is linear in inverse temperature and log time.  This is the movitvation for transforming the time/temperature data as inverse temperature and log time.

$$ \begin{align*}
\ln{\frac{4Dt}{x^2}}&= \ln{D} + \ln{t} + \ln{\frac{4}{x^2}}\\
&= \ln{D_0 e^{-\frac{E_A}{RT}}} + \ln{t} + \ln{\frac{4}{x^2}}\\
& = -\frac{E_A}{R}\frac{1}{T} + \ln{t} +\ln{\frac{4D_0}{x^2}}
\end{align*}$$

This motivation is vindicated as using untransformed data results in a model with negative R-squared values.  This was verified with a smaller model using untransformed (save conversion to Kelvin and minutes along with standard scaling) data which yielded R-squared values of $-0.7$ for temperature and $-0.6$ for time after 20 epochs.

### Data Transformation: Cooling Methods and One Hot Encoding
Cooling data is included in the metadata for each sample.  Diffusion in steel is driven time and temperature so it is reasonable to assume that the cooling method effects the microstructure.  Two samples annealed at 900 °C for 180 minutes may have significant differences in their microstructures if one is furnace cooled and the other is air cooled.  To account for this in the model the cooling methods were one hot encoded for the regressor to predict.

### Transfer Learning: InceptionV3
Transfer learning is a central tool of the model.  InceptionV3 with the top layer removed is used for image processing with inputs scaled to the cropped pictures.  The layers are initially set to untrainable until sufficient converegence for the model is attained.  The training layers are then turned back on to fine tune the model.  

The paper on the architecture of inceptionV3 can be found <a href ="https://arxiv.org/pdf/1512.00567.pdf">here</a>.
### Deep Learning Feed Forward Network

Two hidden layers are used as there is sufficient non-linearity in the problem even with the data transformation.  Batch normalization and dropout were the primary forms of regularization for the model which showed robust resistance to overfitting.  

## Model Results:
### Metrics: R-squared, Mean Absolute Error, Difference Scatter Plot

The model uses Mean Squared Error for optimization, however the two primary metrics for the model are R-squared and Mean Absolute Error.

<table border="1" class="dataframe"align = "center">
  <thead>
    <tr style="text-align: center;">
      <th>Metrics</th>
      <th>Temperature</th>
      <th>Time</th>   </tr>
  </thead>
  <tbody>
    <tr>
      <th>R-Squared</th>
      <td>0.92</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>Mean Absolute Error</th>
      <td>14.8 degrees</td>
      <td>251 minutes</td>
    </tr>
  </tbody>
</table>
<p align = "center">Tabel 1. Metric Data for Model  </p>

We can see that there is better performance of the Temperature variable and we can see in table 1 that most of the temperature errors are with 20 degrees of the true values.   An examanintion of a bullseye diffenece graph between the predicted labels and true labels is due to samples with long anneal times (over 8 hours).  This is reasonable from a physical standpoint as there likely little difference between annealing a sample for 60 hours veruss 85 hours.   This graph can be explored more fully in the model notebook.

<p align="center">
<img src="https://github.com/RobertGWolf/TDI-CapstoneProject/assets/133603510/0b8467fe-8119-4d17-a8d1-3b24c83c86c2" />
<p align = "center">Fig 3.  Difference Scatter Graph between Predicted Time and Temperature and True Time and Temperatures  </p>
</p>.





### Improvements to the Model
Increased access to data for times in the typical heat treatment range (2 to 4 hours) would improve to the model.  There may be improvements to be made by an analysis of the sigularity of the solution to the diffusion equation.  Addiitional use of PIML techniques such as attaching the diffusion equation to the loss function would likely improve the model.  
