
# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

## 1. Dataset y Preprocesamiento
- ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?

Porque el flatten dará vectores de diferentes tamaños para imagenes de diferente resolución. La estructura de la MLP está hecha para un vector de entrada de tamaño constante, por ende si se usan imagenes de diferente tamaño posiblemente haya que hacer un padding o recortar el vector resultante del flatten. Esto puede producir un desplazamiento de la información de la imagen y no concordar con los pesos entrenados de la red o una perdida de información en el caso de recortar el vector.



- ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?

Albumentations ofrece mayor variedad de transformaciones para aumento de datos en computer vision. Albumentations aplica la misma transformación a la imagen y las máscaras que tenga, esto es muy útil en tareas como segmentación donde las anotaciones son cruciales para el ML. Al aplicar la misma transformación para la imagen y la mascara las anotaciones quedan en concordancia para los datos aumentados, por ende Albumentations se puede usar para más tareas, Torchvision transforms no tiene este soporte nativo para máscaras.

- ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?

Normaliza los valores de los pixeles para que estén en un rango de [-1;1]. Normalizar los datos de entrada antes del entrenamiento de una red es importante para un entrenamiento estable. La normalización evita que valores muy grandes o pequeños generen problemas en el aprendizaje por afectar el gradiente en el backpropagation, ayuda evitar exploding gradient y que los valores grandes tengan un peso excesivo en el gradiente de la función de costo. La normalización también contribuye a evitar asimetría en la evolución de los parámetros.

img = (img - mean * max_pixel_value) / (std * max_pixel_value)

- ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?

ToTensorV2 convierte las imágenes de Arrays Numpy en Tensores Pytorch.  Albumentations trabaja con ararys Numpy en formato channel last, mientras que Pytorch con tensores channel first, por esto es necesaria la conversión intermedia.

## 2. Arquitectura del Modelo

- ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?

Usamos una red MLP dado que una CNN es computacionalmente mucho más pesado que una MLP a pesar del ahorro en parámetros. También es un buen acercamiento al flujo de machine learning con un ejemplo abordable y claro de Pytorch.
Una red MLP tiene la limitación de precisar del flatten para interpretar las imagenes, esto pierde mucha riqueza de información por deshacer las relaciones espaciales entre pixeles. Las CNNs en cambio conservan las estructuras espaciales.

- ¿Qué hace la capa `Flatten()` al principio de la red?

La capa flatten convierte las imagenes de un tensor tridimensional a uno unidimensional. Recibe un tensor de 3,64,64 y tiene como output un tensor de (12288,)

- ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?

Se usó la función de activación ReLU. ReLU es una operación más simple y menos costosa computacionalemente ( max(0,x). Asumo que se usa principalmente por el tamaño del vector de entrada. Como hay 12288 features, cada neurona recibe el producto punto de ese vector con sus pesos, los cuales se incializan aleatoriamente cerca del cero. Esto puede facilmente resultar en una salida muy grande o muy chica, que con los valores asintóticos de la sigmoidea/tanh resultaría en salidas de 1 o -1 en casi todas las neuronas, que brinda poca expresividad. En esas colas de la sigmoidea y tanh la derivada es casi nula por ende probablemente habría desvanecimiento de gradiente. En cambio ReLU siempre tiene gradiente 0 o 1, que mitiga el riesgo de tener desvanecimiento de gradiente.

- ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?

Habría que cambiar el parametro input size.

## 3. Entrenamiento y Optimización
- ¿Qué hace `optimizer.zero_grad()`?

En cada pasada del loop de entrenamiento se empieza llamando zero_grad(). Esto inicializa el gradiente en 0 antes del backpropagation, esto se hace porque PyTorch por default acumula los gradientes, los suma en vez de reemplazarlos.  Por ende zero_grad es necesario para que el gradiente calculado dependa del batch 

- ¿Por qué usamos `CrossEntropyLoss()` en este caso?

Usamos CrossEntropyLoss en lugar de la Binary Cross Entropy dado que es un problema multiclase. 

- ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?

El batch size afecta cuan ruidoso es el entrenamiento. A mayor batch size se promedia la loss de más datos para calcular el gradiente en cada actualización, esto hace que el modelo converja con menos ruido. 

- ¿Qué pasaría si no usamos `model.eval()` durante la validación?

Model.eval desactiva comportamientos de capas pertinentes al entrenamiento, como batch normalization o dropout. Sin embargo en este modelo no se usan estas técnicas y por ende la diferencia sería mínima. 

## 4. Validación y Evaluación
- ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?

Significa que hay overfitting a los datos de entrenamiento, el modelo no generaliza y hay que ajustar las técnicas de regularización.

- ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?

En algunos problemas reales la sensibilidad (recall) puede ser más relevant que la precisión (accuracy). Por ejemplo en screening médico el costo de un falso negativo es mucho mayor que el de un falso positivo, porque un falso positivo va ser revisado por un médico mientras que el falso negativo no será detectado y podría evitar una intervención que sería beneficiosa para el paciente.

- ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?

La matriz de confusión nos permite identificar donde se concentran las predicciones erroneas, por ejemplo podemos ver si hay ciertas clases que se confunden entre sí. Esto nos permite tomar una decisión informada respecto a que datos relevar si ampliamos el dataset para mejorar el modelo. Permite evaluar que features podrían agregarse para distinguir las clases que se confunden. 

- En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?

Recall es la sensibilidad de la clasificación, representa el porcentaje de casos positivos que fueron correctamente identificados. En este caso multiclase correspondería al numero ubicado en la diagonal sobre la suma de su fila en la matriz de confusión.

Precisión es el valor predictivo positivo, es decir que porcentaje de las predicciones positivas fueron acertadas. En este caso se puede ver en la matriz de confusión como el valor de la diagonal sobre la suma de su columna. 

F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
Es una media de precisión y recall usada para evaluar el equilibrio entre ambos. 


## 5. TensorBoard y Logging
- ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?

Usar TensorBoard tiene la ventaja de tener una interfaz gráfica para representar la información del entrenamiento y comparar diferentes experimentos. Tiene el beneficio de poder loguear diferentes tipos de datos. 

- ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?

Se anotan tipos de información distintos y van a solapas distintas en la interfaz de TensorBoard.

- ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?

Es útil  porque permite ver trazabilidad en aplicación de transformaciones, confirmar que no hay errores, evaluar por qué puede haber desacuerdo entre métricas de train y val. 

- ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?

Se puede comparar manteniendo un registro de difrentes runs y eligiendo cuales incluir en los gráficos.

## 6. Generalización y Transferencia
- ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?

Habría que cambiar num_classes para que la ultima capa tenga una salida de 100 logits en lugar de 10. Debería volver a armar el dataset y correr el Label Encoder con los nuevos labels para que los logits de salida estén asociados con una clase especifica de las etiquetas. Entonces habría que instanciar nuevamente el modelo con el numero de clases a entrenar correctas e instanciar nuevamente los objetos de los datasets.

- ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?

Una CNN suele ser más adecuada porque realiza operaciones convolucionales sucesivas con kernels (matrices) que luego son recolectados por capas pooling. Esto permite que la red conserve información espacial y aprenda relaciones espaciales. Al resumir estás relaciones espaciales y combinarlas se mejora la clasificación de las imágnees. Esto se debe a que el flatten, para entregar un vector unidimensional a una MLP, producira un vector que indica presencia de elementos espaciales de mayor nivel que intensidades en pixeles desagregados. 

- ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?

El modelo tendría muy pocos datos para la cantidad de parametros que tiene, posiblemente haya overfitting.

- ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?

Unicamente habría que instanciar el modelo con input_size=64*64*1 y adaptar  la clase de datasets para que convierta las imagenes a Numpy leyendolas como greyscale en lugar de RGB. 

## 7. Regularización

### Preguntas teóricas:
- ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?

Es un conjunto de técnicas aplicadas para que un modelo generalice, es decir para evitar el overfitting.

- ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?

Un Dropout Layer consiste en  una capa oculta que inhibe aleatoriamente un porcentaje de una capa de la red. Funciona para evitar la posibilidad de especialización de subredes forzando al modelo a emplear mayor parte de la red evitando dependencia de pesos excesivamente altos. Mientras que weight decay agrega una componente a la función de costo que penaliza el uso de pesos altos. Se llama L2 porque usa la norma L2 del vector de pesos. Ambos métodos reducen el tamaño de los pesos pero lo hacen de maneras fundamentalmente distintas, uno afecta la estructura de la red en train y la otra la función de costo. 

- ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?

El Batch Normalization soluciona el Internal Covariate Shift, un fenómeno que describe como las capas sucesivas ven distribuciones cambiantes por la actualización de pesos de capas anteriores. Este cambio de distribución dificulta el aprendizaje, se describe como un blanco movil, consecuentemente el aprendizaje es menos estable y más lento. 
Para mitigar este fenómeno BatchNorm normaliza luego de cada capa, haciendo la distribución que ve cada capa en su entrada más uniforme a lo largo del entrenamiento, esto permite un aprendizaje más uniforme e iterativo. 

- ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?
 
Al suavizar la convergencia se puede aumentar el learning rate sin tener un aprendizaje con ruido excesivo con posibilidad de no llegar al óptimo de la función de costo. La función de costo se vuelve más simétrica permitiendo este aumento de learning rate.

- ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?

Si bien su objetivo es acelerar el aprendizaje tiene un leve efecto regularizador porque normaliza las salidas de las capas con la media y varianza del batch, este sample pequeño introduce un componente aleatorio que puede brindar un poco de regularización. 

- ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?

Se podría observar las curvas de acuracy o loss en validación. Si hay overfitting el accuracy va reducirse respecto a epochs previos y la loss va aumentar. 

- ¿Cómo ayuda la regularización a mejorar la generalización del modelo?

La regularización mejora la generalización del modelo evitando el overfitting, evita que la red aprenda pesos que se ajustan demasiado a los datos de entrenamiento. Esto lo hace principalmente limitando el tamaño de los pesos dado que los pesos de tamaño excesivo generan fronteras de decisión muy estrictas que no capturan las distribuciones de probabilidad de fenómenos reales. También evita la dominancia de algunos parametros por sobre otros. 

### Actividades de modificación:
1. Agregar Dropout en la arquitectura MLP:
   - Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.
   - Comparar los resultados con y sin `Dropout`.

   El accuracy y loss en train se hicieron mucho más estables, se suavizaron las curvas. Se llega a un valor similar de validation accuracy mucho más rápido. Es interesante que el accuracy en validación es mayor que en train, asumo que es porque en train se usa mitad de la red por el dropout impuesto.




2. Agregar Batch Normalization:
   - Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

     Vemos que aumenta la validation accuracy máxima alcanzada, mientras que el accuracy en train permanece similar. Acuracy en validación se mantiene mayor que en train. 

3. Aplicar Weight Decay (L2):
   - Modificar el optimizador:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
     ```

     Observamos que el train accuracy mejora considerablemente, los valores en validación permanencen casi igaules

4. Reducir overfitting con data augmentation:
   - Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.

   Se emplearon desde le comienzo con train_transform que emplea estás 3 transformaciones.

5. Early Stopping (opcional):
   - Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.

### Preguntas prácticas:
- ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?

BatchNorm aceleró notablemente el entrenamiento sin embargo parece bajar la estabilidad. Esto puede ser efecto de alteraciones de escala de los gráficos dado que se esperaría que aumente la estabilidad por normalizar los outputs de las capas.

- ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?
Si, mejoró. Se alcanzó accurac en validación más alta.

- ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?


- ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?

Disminuyó considerablemente, lo más llamativo es la diminución del valor inicial. Atribuyo esta caida en el valor inicial a la reducción de la importancia de los pesos de inicialización producto de reducir el Internal Covariate Shift.
## 8. Inicialización de Parámetros

### Preguntas teóricas:
- ¿Por qué es importante la inicialización de los pesos en una red neuronal?

Es importante sobretood porque deseamos empezar en una región de la función de activación con un gradiente útil, por ejemplo si se empieza en la sigmoidea sobre una asintota seguramente el entrenamiento sufrirá de vanishing gradient.

- ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?

Podrían producirse simetrías que no se rompan y se pierda mucha de la capacidad de generalización de la red.

- ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?

EL criterio de Glorot fija que la varianza de salida de una capa debe ser igual a la varianza de entrada para mantener la energía constante através de esa capa. Se usa priuncipalmente para funciones de activación simétricas como sigmoidea o tanh. He se usa con el mismo propósito para funciones de activación asimétricas como ReLU, para mantener la varianza se escala la función para compensar su varianza reducida.

- ¿Por qué en una red con ReLU suele usarse la inicialización de He?

Porque Hu usa una varianza proporcional a 2/n, compensando la rama anulada por ReLU mientras que GLorot usa 1/n que resultaría en una reducción de varianza.

- ¿Qué capas de una red requieren inicialización explícita y cuáles no?

Las capas ocultas y de salida, la capa de entrada no. 

### Actividades de modificación:
1. Agregar inicialización manual en el modelo:
   - En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
     ```python
     def init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.zeros_(m.bias)
     ```

   Mejoran mucho las accuracies alcanzadas tanto en train como en val, empieza a notarse efecto de overfitting.

2. Probar distintas estrategias de inicialización:
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)

   Debería mejorar performance respecto de Xavier dado que las funciones de activación usadas son ReLU. Efectivamente se ve un gran aumento del accuracy en train usando la inicialización de He. 
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.

   La velocidad se ve muy beneficiada por un punto de partida mucho mejor que los anteriores, la loss empieza con valores notablemente más chicos y el accuracy es más alto desde el comienzo. La estabilidad es dificil de observar en TensorBoard, esperaría que sea más estable pero honestamente no lo puedo discernir para Xavier. Al contrario con He se observa estabilidad mucho mayor, principalmente por el rango de la escala y no en la forma de la evolución del trazo.

3. Visualizar pesos en TensorBoard:
   - Agregar esta línea en la primera época para observar los histogramas:
     ```python
     for name, param in model.named_parameters():
         writer.add_histogram(name, param, epoch)
     ```

### Preguntas prácticas:
- ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?

- ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?

No.

- ¿Qué impacto tiene la inicialización sobre las métricas de validación?

 Influye sobretodo en el punto de partida.

- ¿Por qué `bias` se suele inicializar en cero?

Como el bias es independiente al resto de pesos, no afecta la simetría por ende no es necesario inicializarlo aleatoriamente.