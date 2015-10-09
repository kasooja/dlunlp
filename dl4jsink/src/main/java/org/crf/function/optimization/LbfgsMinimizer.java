package org.crf.function.optimization;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.ListIterator;

import org.apache.log4j.Logger;
import org.crf.function.DerivableFunction;
import org.crf.utilities.DerivableFunctionWithLastCache;
import org.crf.utilities.CrfException;
import org.crf.utilities.VectorUtilities;

/**
 * Implementation of L-BFGS algorithm for minimizing a function.
 * <BR>
 * L-BFGS stands for "Limited memory BFGS", where "BFGS" is an acronym of
 * "Broyden Fletcher Goldfarb Shanno" who developed the BFGS algorithm.
 * <BR>
 * The BFGS algorithm approximates Newton method for optimization, by approximating
 * the inverse of the Hessian without calculating the exact Hessian.
 * The L-BFGS algorithm approximates the BFGS algorithm by approximating calculations
 * that are performed with the inverse of the Hessian, but stores neither the
 * inverse of the Hessian nor its approximation in the memory.
 * Thus the L-BFGS algorithm is much cheaper in space complexity notion.
 * <BR>
 * The L-BFGS algorithm is described in the book "Numerical Optimization" by Jorge Nocedal and Stephen J. Wright,
 * Chapter 9. The book can be downloaded from http://www.bioinfo.org.cn/~wangchao/maa/Numerical_Optimization.pdf 
 *  
 *  
 * 
 * @author Asher Stern
 * Date: Nov 7, 2014
 *
 */
public class LbfgsMinimizer extends Minimizer<DerivableFunction>
{
	public static final int DEFAULT_NUMBER_OF_PREVIOUS_ITERATIONS_TO_MEMORIZE = 20;
	public static final double DEFAULT_CONVERGENCE = 5000; //0.001

	public LbfgsMinimizer(DerivableFunction function)
	{
		this(function,DEFAULT_NUMBER_OF_PREVIOUS_ITERATIONS_TO_MEMORIZE,DEFAULT_CONVERGENCE);
	}

	public LbfgsMinimizer(DerivableFunction function, int numberOfPreviousIterationsToMemorize, double convergence)
	{
//		super(function);
		super(new DerivableFunctionWithLastCache(function));
		this.numberOfPreviousIterationsToMemorize = numberOfPreviousIterationsToMemorize;
		this.convergence = convergence;
	}

	@Override
	public void find()
	{
		previousItrations = new LinkedList<PointAndGradientSubstractions>();
		LineSearch<DerivableFunction> lineSearch = new ArmijoLineSearch<DerivableFunction>();
		
		point = new double[function.size()];
		for (int i=0;i<point.length;++i) {point[i]=0.0;}
		value = function.value(point);
		if (logger.isInfoEnabled()) {logger.info("LBFGS: initial value = "+String.format("%-3.3f", value));}
		double[] gradient = function.gradient(point);
		double previousValue = value;
		int forLogger_iterationIndex=0;
		do
		{
			previousValue = value;
			double[] previousPoint = point;
			double[] previousGradient = gradient;
			
			double[] direction = VectorUtilities.multiplyByScalar(-1.0, twoLoopRecursion(point));
			double alpha_rate = lineSearch.findRate(function, point, direction);
			point = VectorUtilities.addVectors(point, VectorUtilities.multiplyByScalar(alpha_rate, direction));
			value = function.value(point);
			gradient = function.gradient(point);
			
			previousItrations.add(new PointAndGradientSubstractions(VectorUtilities.substractVectors(point, previousPoint), VectorUtilities.substractVectors(gradient, previousGradient)));
			if (previousItrations.size()>numberOfPreviousIterationsToMemorize)
			{
				previousItrations.removeLast();
			}
			if (previousItrations.size()>numberOfPreviousIterationsToMemorize) {throw new CrfException("BUG");}
			
			if (value>previousValue) {logger.error("LBFGS: value > previous value");}
			++forLogger_iterationIndex;
			if (logger.isInfoEnabled()) {logger.info("LBFGS iteration "+forLogger_iterationIndex+": value = "+String.format("%-3.3f", value));}
		}
		while(Math.abs(previousValue-value)>convergence);
		
		calculated = true;
	}

	@Override
	public double getValue()
	{
		if (!calculated) {throw new CrfException("Not calculated.");}
		return value;
	}

	@Override
	public double[] getPoint()
	{
		if (!calculated) {throw new CrfException("Not calculated.");}
		return point;
	}
	
	
	
	private double[] twoLoopRecursion(double[] point)
	{
		ArrayList<Double> rhoList = new ArrayList<Double>(previousItrations.size());
		ArrayList<Double> alphaList = new ArrayList<Double>(previousItrations.size());
		
		double[] q = function.gradient(point);
		for (PointAndGradientSubstractions substractions : previousItrations)
		{
			double rho = 1.0/VectorUtilities.product(substractions.getGradientSubstraction(), substractions.getPointSubstraction());
			rhoList.add(rho);
			double alpha = rho*VectorUtilities.product(substractions.getPointSubstraction(), q);
			alphaList.add(alpha);
			
			q = VectorUtilities.substractVectors(q, VectorUtilities.multiplyByScalar(alpha, substractions.getGradientSubstraction()) );
		}
		
		double[] r = calculateInitial_r_forTwoLoopRecursion(q);

		ListIterator<PointAndGradientSubstractions> previousIterationsIterator = previousItrations.listIterator(previousItrations.size());
		ListIterator<Double> rhoIterator = rhoList.listIterator(rhoList.size());
		ListIterator<Double> alphaIterator = alphaList.listIterator(alphaList.size());
		while (previousIterationsIterator.hasPrevious()&&rhoIterator.hasPrevious()&&alphaIterator.hasPrevious())
		{
			PointAndGradientSubstractions substractions = previousIterationsIterator.previous();
			double rho = rhoIterator.previous();
			double alpha = alphaIterator.previous();
			
			double betta = rho * VectorUtilities.product(substractions.getGradientSubstraction(), r);
			r = VectorUtilities.addVectors( r, VectorUtilities.multiplyByScalar(alpha-betta, substractions.getPointSubstraction()) );
		}
		if ((previousIterationsIterator.hasPrevious()||rhoIterator.hasPrevious()||alphaIterator.hasPrevious())) {throw new CrfException("BUG");}
		
		return r;
	}
	
	
	private double[] calculateInitial_r_forTwoLoopRecursion(double[] q)
	{
		double gamma = 1.0;
		if (previousItrations.size()>=1)
		{
			PointAndGradientSubstractions lastSubstraction = previousItrations.get(0);
			gamma =
					VectorUtilities.product(lastSubstraction.getPointSubstraction(), lastSubstraction.getGradientSubstraction())
					/
					VectorUtilities.product(lastSubstraction.getGradientSubstraction(), lastSubstraction.getGradientSubstraction());
			
			
		}
		
		double[] r = VectorUtilities.multiplyByScalar(gamma, q);
		return r;
	}
	

	

	
	
	private final int numberOfPreviousIterationsToMemorize; // m
	private final double convergence;
	
	private LinkedList<PointAndGradientSubstractions> previousItrations; // newest is pushed to the beginning.
	
	private double[] point = null;
	private double value = 0.0;
	
	private boolean calculated = false;
	


	private static final Logger logger = Logger.getLogger(LbfgsMinimizer.class);
}
