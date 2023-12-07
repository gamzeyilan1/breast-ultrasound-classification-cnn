classdef Restoration < featlearn.FeatureTransform
    properties(GetAccess = public, SetAccess = protected, Dependent)
        NumPredictors;
        NumLearnedFeatures;
        Mu;
        Sigma;
        FitInformation;
        TransformWeights;
        InitialTransformWeights;
        NonGaussianity;
    end
    
    methods
        function fitInfo = get.FitInformation(this)
            fitInfo = this.Impl.FitInformation;
        end
        
        function transformWeights = get.TransformWeights(this)
            transformWeights = this.Impl.TransformWeights;
        end
        
        function initialTransformWeights = get.InitialTransformWeights(this)
            initialTransformWeights = this.Impl.InitialTransformWeights;
        end
        
        function nonGaussianity = get.NonGaussianity(this)
            nonGaussianity = this.Impl.NonGaussianity;
        end
        
        function mu = get.Mu(this)
            mu = this.Impl.Mu;
        end
        
        function sigma = get.Sigma(this)
            sigma = this.Impl.Sigma;
        end
        
        function numPredictors = get.NumPredictors(this)
            numPredictors = this.Impl.NumPredictors;
        end
        
        function numLearnedFeatures = get.NumLearnedFeatures(this)
            numLearnedFeatures = this.Impl.NumLearnedFeatures;
        end
    end
    
    methods(Hidden)
        function this = ReconstructionICAUpdated(X, Q, varargin)
            X = featlearn.utils.InputValidator.validateX(X);
            Q = featlearn.utils.InputValidator.validateQ(Q);
            X = featlearn.utils.InputValidator.removeBadRows(X, []);
            
            if isempty(X)
                error(message('stats:featlearn:ReconstructionICAUpdated:EmptyPredictors'));
            end
            
            defaultInitialTransformWeights = [];
            defaultNonGaussianity = ones(Q, 1);
            paramNames = {'InitialTransformWeights', 'NonGaussianity'};
            paramDefaults = {defaultInitialTransformWeights, defaultNonGaussianity};
            
            [initialTransformWeights, nonGaussianity, ~, otherArgs] = internal.stats.parseArgs(paramNames, paramDefaults, varargin{:});
            
            if ~isempty(initialTransformWeights)
                P = size(X, 2);
                initialTransformWeights = featlearn.utils.InputValidator.validateInitialTransformWeights(initialTransformWeights, P, Q);
            end
            
            nonGaussianity = featlearn.utils.InputValidator.validateNonGaussianityIndicator(nonGaussianity, Q);
            
            modelParams = featlearn.params.ReconstructionICAParams(X, Q, otherArgs{:});
            
            this.Impl = featlearn.impl.ReconstructionICAImpl(X, Q, initialTransformWeights, nonGaussianity, modelParams);
        end
    end
    
    methods
        function Z = transform(this, X, varargin)
            X = featlearn.utils.InputValidator.validateX(X);
            [X, ~, badRows] = featlearn.utils.InputValidator.removeBadRows(X, []);
            
            if isempty(X)
                error(message('stats:featlearn:ReconstructionICAUpdated:EmptyPredictors'));
            end
            
            P = this.NumPredictors;
            
            if size(X, 2) ~= P
                error(message('stats:featlearn:ReconstructionICAUpdated:PredictorSizeMismatch', P));
            end
            
            Q = this.NumLearnedFeatures;
            Z = nan(length(badRows), Q);
            Z(~badRows, :) = transform(this.Impl, X);
        end
    end
    
    methods(Hidden)
        function s = propsForDisp(this, s)
            s = propsForDisp@featlearn.FeatureTransform(this, s);
            
            s.NumPredictors = this.NumPredictors;
            s.NumLearnedFeatures = this.NumLearnedFeatures;
            s.Mu = this.Mu;
            s.Sigma = this.Sigma;
            s.FitInformation = this.FitInformation;
            s.TransformWeights = this.TransformWeights;
            s.InitialTransformWeights = this.InitialTransformWeights;
            s.NonGaussianity = this.NonGaussianity;
        end
    end
end
