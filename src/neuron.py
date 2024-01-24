# Importamos las librerias
import numpy as np
import math

# Inicializamos la clase Neuron
class Neuron:

  def __init__(self, weights=[0], bias=0, func="relu"):
    # Constructor de Neuron con las diferentes propiedades
    self.weights = weights
    self.bias = bias
    self.func = func


  @staticmethod
  def __relu(x):
    # Metodo estatico para la funcion de activacion relu
    return max(0, x)

  @staticmethod
  def __sigmoide(x):
    # Metodo estatico para la funcion de activacion sigmoide
    return 1 / (1 + math.exp(-x))

  @staticmethod
  def __tanh(x):
    # Metodo estatico para la funcion de activacion tanh
    return math.tanh(x)

  def run(self, input_data=[]):
    # Creamos la funcion run() para calcular el resultado

    # Comprobamos que los pesos y las entradas tienen la misma longitud
    if len(self.weights) == len(input_data):

      x = np.dot(self.weights, input_data) + self.bias

      funcion = getattr(Neuron, f"_Neuron__{self.func.lower()}")

      return funcion(x)

    else:
      # Si los pesos y las entradas no son de la misma longitud, devuelve un mensaje de error
      return "Los pesos y los datos de entrada no tienen la misma longitud"

  # Funcion para cambiar el valor del bias (b)
  def change_bias(self, bias):
    self.bias = bias

  # Funcion para cambiar el valor de los pesos (w)
  def change_weights(self, w):
    if len(w) == len(self.weights):
      self.weights = w
    else:
      print( f"Los pesos y los datos de entrada no tienen la misma longitud. Los pesos se quedan con los valores: {self.weights}")
