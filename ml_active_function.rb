# Activition Functions

class MLActiveFunction
  def tanh(x)
    ( 2.0 / ( 1.0 + Math.exp(-1.0 * x) ) ) - 1.0
    # return Math.tanh(_x)
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

  def dashTanh(output)
    ( 1.0 - ( output * output ) ) * 0.5
  end

  def dashSigmoid(output)
    ( 1.0 - output ) * output
  end

  def dashSgn(output)
    output
  end

  def dashRbf(output)
    output
    #return 1.0 + _output + ( _output * _output )
    #return 1.0 - _output
  end
end
