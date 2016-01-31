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

	def initialize()
		@patterns = Array.new
		@weights  = Array.new
		@targets  = Array.new
	end

	def self.sharedDelta
		return @@sharedDelta
	end

	# Public methods
	def addPatterns(inputs, target)
		@patterns.push(inputs)
		@targets.push(target)
	end

	def setupWeights(weights)
		if @weights.count > 0
			@weights.clear()
		end
		@weights.push(weights)
		puts "weights : #{weights}"
	end

	def randomWeights()
	
		if @weights.count > 0
			@weights.clear()
		end
		# Follows the inputs count to decide how many weights it needs.
		_randomMaker   = Random.new
		_inputNetCount = @patterns.count
		_inputMax 	   = DEFAULT_RANDOM_MAX / _inputNetCount
		_inputMin	   = DEFAULT_RANDOM_MIN / _inputNetCount
		for i in (0.._inputNetCount)
			@weights.push(_randomMaker.rand(_inputMin.._inputMax))
		end
	end

	def training()

	end

	def trainingWithCompletion(&block)

		if block_given?
			block.call(true, [1, 2], 99);
		end
		
	end

	def trainingWithIteration(iterationBlock, completionBlock)

		if !iterationBlock.nil?
			iterationBlock.call(1, [0.1, 0.2, -0.5])
		end

		if !completionBlock.nil?
			completionBlock.call(true, [3, 4], 99)
		end

	end

	def directOutputByPatterns(p, &block)

		if block_given?
			block.call([1.0])
		end

	end



end

delta = MLDelta.new
delta.addPatterns([1.0, 2.0, 3.0], 1.0)
delta.addPatterns([5.0, 6.0, 7.0], -1.0)
#delta.setupWeights([0.4, 0.5, -0.1])
delta.randomWeights()

delta.trainingWithCompletion { 
	|a, b, c| 
	puts "hello #{a} #{b} #{c}" 
}

iterationBlock = Proc.new do |iteration, weights|
	puts "iteration : #{iteration}, weights : #{weights}"
end

completionBlock = Proc.new do |success, weights, totalIteration|
	puts "success : #{success}, weights : #{weights}, totalIteration : #{totalIteration}"
end

delta.trainingWithIteration(iterationBlock, completionBlock)

