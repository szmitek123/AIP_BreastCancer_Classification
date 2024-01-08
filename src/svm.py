import numpy as np

from typing import Callable
from kernels.rbf import RBF

# Support Vector Machine
class SVM:

    # Constructor
    def __init__(self, kernel: Callable[[ np.ndarray, np.ndarray ], np.float64 ] = RBF(gamma=1), coherence: np.float64 = 1, tolerance: np.float64 = 1e-4, numEpochs: int = 100, numPasses: int = 10) -> None:
        self.kernel = kernel
        self.coherence = coherence
        self.tolerance = tolerance
        self.numEpochs = numEpochs
        self.numPasses = numPasses

        # Init empty external parameters
        self.inputs: np.ndarray = np.empty([ 0, 0 ])
        self.expect: np.ndarray = np.empty([ 0 ])

        # Init empty internal parameters
        self.a: np.ndarray = np.empty([ 0 ])
        self.b: np.float64 = 0.0

    # Calculate margin
    def margin(self, input: np.ndarray) -> np.float64:
        return np.sum(self.a * self.expect * self.kernel(input, self.inputs)) + self.b
    
    # Calculate prediction vector
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        predict: np.ndarray = np.empty([ inputs.shape[0] ])

        for i in range(0, inputs.shape[0]):
            predict[i] = self.margin(inputs[i])

        return np.where(predict > 0, 1, -1)
        
    # Calculate border
    def train(self, inputs: np.ndarray, expect: np.ndarray) -> None:
        # Save external parameters
        self.inputs = inputs
        self.expect = expect

        # Init internal parameters
        self.a = np.zeros([ self.inputs.shape[0] ])
        self.b = 0.0

        # Init training parameters
        epochs = 0
        passes = 0

        while epochs < self.numEpochs and passes < self.numPasses:
            adjusted = False

            for i in range(0, self.inputs.shape[0]):
                adjusted = SMO(self, i).adjust()

            # Update counter
            epochs += 1
            passes += 1

            # Reset counter
            if adjusted: passes = 0

# Sequential Minimal Optimization
class SMO:

    # Constructor
    def __init__(self, svm: SVM, i: int) -> None:
        self.svm = svm
        self.indexI = i
        self.indexJ = np.random.randint(0, svm.inputs.shape[0] - 1)

        # Exclude i == j
        if self.indexJ >= self.indexI: self.indexJ += 1

        # Init internal parameters
        self.kernel: np.float64 = 0
        self.boundM: np.float64 = 0
        self.boundN: np.float64 = self.svm.coherence
        self.errorI: np.float64 = self.svm.margin(self.svm.inputs[self.indexI]) - self.svm.expect[self.indexI]
        self.errorJ: np.float64 = self.svm.margin(self.svm.inputs[self.indexJ]) - self.svm.expect[self.indexJ]

    # Calculate offset & adjust
    def adjust(self) -> bool:
        if (self.passesStep1()): return False
        if (self.passesStep2()): return False

        # Shortcuts
        AI = self.svm.a[self.indexI]
        AJ = self.svm.a[self.indexJ]
        C = self.svm.coherence
        I = self.indexI
        J = self.indexJ
        M = self.boundM
        N = self.boundN

        # calculate new AI value
        newAJ = AJ - (self.svm.expect[J] * (self.errorI - self.errorJ)) / self.kernel
        if (newAJ > N): newAJ = N
        if (newAJ < M): newAJ = M

        # guard - test numerical tolerance
        if (np.abs(AJ - newAJ) < self.svm.tolerance): return False

        # calculate new AI value
        newAI = AI + self.svm.expect[I] * self.svm.expect[J] * (AJ - newAJ)
        self.svm.a[J] = newAJ
        self.svm.a[I] = newAI

        # calculate new bias value
        b1 = self.getNewB1(newAI - AI, newAJ - AJ)
        b2 = self.getNewB2(newAI - AI, newAJ - AJ)

        # update bias
        self.svm.B = (b1 + b2) * 0.5

        # clamp bias value
        if (newAI > 0 and newAI < C): self.svm.b = b1
        if (newAJ > 0 and newAJ < C): self.svm.b = b2

        return True

    def getNewB1(self, deltaI: np.float64, deltaJ: np.float64) -> np.float64:
        # Shortcuts
        I = self.indexI
        J = self.indexJ

        # Compute partials
        edgeI = self.svm.expect[I] * deltaI * self.svm.kernel(self.svm.inputs[I], self.svm.inputs[I])
        edgeJ = self.svm.expect[J] * deltaJ * self.svm.kernel(self.svm.inputs[I], self.svm.inputs[J])

        return self.svm.b - self.errorI - edgeI - edgeJ

    def getNewB2(self, deltaI: np.float64, deltaJ: np.float64) -> np.float64:
        # Shortcuts
        I = self.indexI
        J = self.indexJ

        # Compute partials
        edgeI = self.svm.expect[I] * deltaI * self.svm.kernel(self.svm.inputs[I], self.svm.inputs[J])
        edgeJ = self.svm.expect[J] * deltaJ * self.svm.kernel(self.svm.inputs[J], self.svm.inputs[J])

        return self.svm.b - self.errorJ - edgeI - edgeJ

    # Calculate step 1
    def passesStep1(self) -> bool:
        # Shortcuts
        C = self.svm.coherence
        I = self.indexI
        
        return not (
            (self.errorI * self.svm.expect[I] < -self.svm.tolerance and self.svm.a[I] < C) or
            (self.errorI * self.svm.expect[I] > +self.svm.tolerance and self.svm.a[I] > 0))

    # Calculate step 2
    def passesStep2(self) -> bool:
        # Shortcuts
        AI = self.svm.a[self.indexI]
        AJ = self.svm.a[self.indexJ]
        C = self.svm.coherence
        I = self.indexI
        J = self.indexJ

        # Find adjust bounds
        if (self.svm.expect[I] == self.svm.expect[J]):
            self.boundM = np.max([0.0, AI + AJ - C])
            self.boundN = np.min([C, AI + AJ])
        else:
            self.boundM = np.max([0.0, AJ - AI])
            self.boundN = np.min([C, AJ - AI + C])

        # Guard - test numerical tolerance
        if (np.abs(self.boundM - self.boundN) < self.svm.tolerance): return True

        # Compute subkernels
        kernelI = self.svm.kernel(self.svm.inputs[I], self.svm.inputs[I])
        kernelJ = self.svm.kernel(self.svm.inputs[J], self.svm.inputs[J])
        kernelX = self.svm.kernel(self.svm.inputs[I], self.svm.inputs[J])

        # Memorize
        self.kernel = 2 * kernelX - kernelI - kernelJ

        return self.kernel >= 0
