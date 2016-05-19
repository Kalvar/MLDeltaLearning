# Activition Functions

class MLActiveFunction
  def tanh(x)
    ( 2.0 / ( 1.0 + Math.exp(-1.0 * x) ) ) - 1.0
  end

  def sigmoid(x)
    1.0 / ( 1.0 + Math.exp(-1.0 * x) )
  end

  def sgn(x)
    ( x >= 0.0 ) ? 1.0 : -1.0
  end

  def rbf(x, sigma)
    Math.exp((-x) / (2.0 * sigma * sigma))
  end

  def dash_tanh(output)
    ( 1.0 - ( output * output ) ) * 0.5
  end

  def dash_sigmoid(output)
    ( 1.0 - output ) * output
  end

  def dash_sgn(output)
    output
  end

  def dash_rbf(output, sigma)
    -((2.0 * output) / (2.0 * sigma * sigma)) * Math.exp((-output) / (2.0 * sigma * sigma))
  end
end
