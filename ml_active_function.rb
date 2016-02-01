# Activition Functions

class MLActiveFunction
  def initialize
    
  end
  
  public
  def tanh(_x)
    return ( 2.0 / ( 1.0 + Math.exp(-1.0 * _x) ) ) - 1.0
    # return Math.tanh(_x)
  end
  
  def sigmoid(_x)
    return ( 1.0 / ( 1.0 + Math.exp(-1.0 * _x) ) )
  end
  
  def sgn(_x)
    return ( _x >= 0.0 ) ? 1.0 : -1.0
  end
  
  def dashTanh(_output)
    return ( 1.0 - ( _output * _output ) ) * 0.5
  end
  
  def dashSigmoid(_output)
    return ( 1.0 - _output ) * _output
  end

  def dashSgn(_output)
    return _output
  end
  
end