require './ml_active_function'
require './ml_active_method'

DEFAULT_RANDOM_MAX = 0.5
DEFAULT_RANDOM_MIN = -0.5

class MLDelta
  @@sharedDelta = MLDelta.new
  attr_accessor :patterns
  attr_accessor :weights
  attr_accessor :targets
  attr_accessor :learningRate
  attr_accessor :maxIteration
  attr_accessor :convergenceValue
  attr_accessor :activeMethod

  attr_accessor :iterationBlock
  attr_accessor :completionBlock

  def initialize
    @active_function = MLActiveFunction.new
    @iteration       = 0
    @sum_error        = 0.0

    @patterns         = []
    @weights          = []
    @targets          = []
    @learningRate     = 0.5
    @maxIteration     = 1
    @convergenceValue = 0.001
    @activeMethod     = MLActiveMethod::TANH

    @iterationBlock   = nil
    @completionBlock  = nil
  end

  # Public methods
  public

  def self.sharedDelta
    return @@sharedDelta
  end

  def addPatterns(inputs, target)
    @patterns << inputs
    @targets << target
  end

  def setupWeights(weights)
    @weights.clear if @weights.count > 0
    @weights += weights
  end

  def randomWeights
    if @weights.count > 0
      @weights.clear()
    end

    # Follows the inputs count to decide how many weights it needs.
    _randomMaker   = Random.new
    _inputNetCount = @patterns.first().count
    _inputMax      = DEFAULT_RANDOM_MAX / _inputNetCount
    _inputMin      = DEFAULT_RANDOM_MIN / _inputNetCount
    for i in (0..._inputNetCount)
      @weights.push(_randomMaker.rand(_inputMin.._inputMax))
    end
  end

  def training
    @iteration += 1
    @sum_error = 0.0
    @patterns.each_with_index{ |inputs, patternIndex| _turningWeightsWithInputs(inputs, @targets[patternIndex]) }

    if (@iteration >= @maxIteration) || (_calculateIterationError() <= @convergenceValue)
      @completionBlock.call( true, @weights, @iteration ) unless @completionBlock.nil?
    else
      @iterationBlock.call( @iteration, @weights ) unless @iterationBlock.nil?
      training
    end
  end

  def trainingWithCompletion(&_block)
    @completionBlock = _block
    training
  end

  def trainingWithIteration(iterationBlock, completionBlock)
    @iterationBlock = iterationBlock
    @completionBlock = completionBlock
    training
  end

  def directOutputByPatterns(inputs, &block)
    block.call(_fOfNetWithInputs(inputs)) if block_given?
  end

  private
  def multiply_matrix(matrix, number)
    matrix.map{ |obj| obj * number }
  end

  def _plusMatrix(_matrix, _anotherMatrix)
    _matrix.collect.with_index{ |obj, i| obj + _anotherMatrix[i] }
  end

  def _activateOutputValue(net_output)
    case @activeMethod
      when MLActiveMethod::SGN
        @active_function.sgn(net_output)
      when MLActiveMethod::SIGMOID
        @active_function.sigmoid(net_output)
      when MLActiveMethod::TANH
        @active_function.tanh(net_output)
      when MLActiveMethod::RBF
        @active_function.rbf(net_output, 2.0)
      else
        # Nothing else
        net_output
    end
  end

  def _fOfNetWithInputs(_inputs)
    # to_f
    _sum = 0.0
    _inputs.each.with_index{ |obj, i| _sum += ( obj * @weights[i] ) }
    return _activateOutputValue(_sum)
  end

  def _fDashOfNet(net_output)
    case @activeMethod
      when MLActiveMethod::SGN
        @active_function.dashSgn(net_output)
      when MLActiveMethod::SIGMOID
        @active_function.dashSigmoid(net_output)
      when MLActiveMethod::TANH
        @active_function.dashTanh(net_output)
      when MLActiveMethod::RBF
        #_@active_function.dashRbf(net_output)
      else
        # Nothing else
        net_output
    end
  end

  # Delta defined cost function formula
  def _calculateIterationError
    (@sum_error / @patterns.count) * 0.5
  end

  def sum_error(error_value)
    @sum_error += (error_value ** 2)
  end

  def _turningWeightsWithInputs(inputs, target_value)
    _weights      = @weights
    _netOutput    = _fOfNetWithInputs(inputs)
    _dashOutput   = _fDashOfNet(_netOutput)
    _errorValue   = target_value - _netOutput

    # new weights = learning rate * (target value - net output) * f'(net) * x1 + w1
    _sigmaValue   = @learningRate * _errorValue * _dashOutput
    _deltaWeights = multiply_matrix(inputs, _sigmaValue)
    _newWeights   = _plusMatrix(_weights, _deltaWeights)

    @weights.clear()
    @weights += _newWeights
    sum_error(_errorValue)
  end

end

# Use methods
delta                  = MLDelta.new
delta.activeMethod     = MLActiveMethod::TANH
delta.learningRate     = 0.8
delta.convergenceValue = 0.001
delta.maxIteration     = 1000
delta.addPatterns([1.0, -2.0, 0.0, -1.0], -1.0)
delta.addPatterns([0.0, 1.5, -0.5, -1.0], 1.0)
delta.setupWeights([1.0, -1.0, 0.0, 0.5])
#delta.randomWeights()

iterationBlock = Proc.new do |iteration, weights|
  puts "iteration : #{iteration}, weights : #{weights}"
end

completionBlock = Proc.new do |success, weights, totalIteration|
  puts "success : #{success}, weights : #{weights}, totalIteration : #{totalIteration}"
  delta.directOutputByPatterns([1.0, -2.0, 0.0, -1.0]){ |predication| puts "predication result is #{predication}" }
end

delta.trainingWithIteration(iterationBlock, completionBlock)

# delta.trainingWithCompletion {
#   |success, weights, totalIteration|
#   puts "success : #{success}, weights : #{weights}, totalIteration : #{totalIteration}"
# }
