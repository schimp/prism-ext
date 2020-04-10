//==============================================================================
//	
//	Copyright (c) 2014-
//	Authors:
//	* Xueyi Zou <xz972@york.ac.uk> (University of York)
//	* Dave Parker <d.a.parker@cs.bham.ac.uk> (University of Birmingham)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================

package explicit;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import explicit.rewards.MDPRewards;
import explicit.rewards.MDPRewardsSimple;
import prism.Pair;
import prism.PrismComponent;
import prism.PrismException;
import prism.PrismNotSupportedException;
import prism.PrismUtils;

/**
 * Explicit-state model checker for partially observable Markov decision processes (POMDPs).
 */
public class POMDPModelChecker extends ProbModelChecker
{
	/**
	 * Create a new POMDPModelChecker, inherit basic state from parent (unless null).
	 */
	public POMDPModelChecker(PrismComponent parent) throws PrismException
	{
		super(parent);
	}

	/**
	 * Compute reachability/until probabilities.
	 * i.e. compute the min/max probability of reaching a state in {@code target},
	 * while remaining in those in @{code remain}.
	 * @param pomdp The POMDP
	 * @param remain Remain in these states (optional: null means "all")
	 * @param target Target states
	 * @param min Min or max probabilities (true=min, false=max)
	 */
	public ModelCheckerResult computeReachProbs(POMDP pomdp, BitSet target, boolean min) throws PrismException
	{
		ModelCheckerResult res = null;
		long timer;
		String stratFilename = null;

		// Check for multiple initial states 
		if (pomdp.getNumInitialStates() > 1) {
			throw new PrismNotSupportedException("POMDP model checking does not yet support multiple initial states");
		}
		
		// Start probabilistic reachability
		timer = System.currentTimeMillis();
		mainLog.println("\nStarting probabilistic reachability (" + (min ? "min" : "max") + ")...");

		// If required, create/initialise strategy storage
		if (genStrat || exportAdv) {
			stratFilename = exportAdvFilename;//"policyGraph.txt";
		}

		// Compute rewards
		res = computeReachProbsFixedGrid(pomdp, target, min, stratFilename);

		// Finished probabilistic reachability
		timer = System.currentTimeMillis() - timer;
		mainLog.println("Probabilistic reachability took " + timer / 1000.0 + " seconds.");

		// Update time taken
		res.timeTaken = timer / 1000.0;
		return res;
	}

	/**
	 * Compute expected reachability rewards using Lovejoy's fixed-resolution grid approach.
	 * Optionally, store optimal (memoryless) strategy info. 
	 * @param pomdp The POMMDP
	 * @param mdpRewards The rewards
	 * @param target Target states
	 * @param inf States for which reward is infinite
	 * @param min Min or max rewards (true=min, false=max)
	 * @param strat Storage for (memoryless) strategy choice indices (ignored if null)
	 */
	protected ModelCheckerResult computeReachProbsFixedGrid(POMDP pomdp, BitSet target, boolean min, String stratFilename) throws PrismException
	{
		if (!(pomdp instanceof POMDPSimple)) {
			throw new PrismException("Sorry, FixedGrid does not support POMDP other than POMDPSimple.");
		}
		POMDPSimple simplePOMDP = (POMDPSimple) pomdp;

		// Start fixed-resolution grid approximation
		long timer = System.currentTimeMillis();
		mainLog.println("Starting fixed-resolution grid approximation (" + (min ? "min" : "max") + ")...");

		// Find out the observations for the target states
		TreeSet<Integer> targetObservsSet = new TreeSet<>();
		for (int bit = target.nextSetBit(0); bit >= 0; bit = target.nextSetBit(bit + 1)) {
			targetObservsSet.add(simplePOMDP.getObservation(bit));
		}
		LinkedList<Integer> targetObservs = new LinkedList<>(targetObservsSet);

		// Initialise the grid points
		ArrayList<Belief> gridPoints = new ArrayList<>();//the set of grid points (discretized believes)
		ArrayList<Belief> unknownGridPoints = new ArrayList<>();//the set of unknown grid points (discretized believes)
		initialiseGridPoints(simplePOMDP, targetObservs, gridPoints, unknownGridPoints);
		int unK = unknownGridPoints.size();
		mainLog.print("Grid statistics: resolution=" + gridResolution);
		mainLog.println(", points=" + gridPoints.size() + ", unknown points=" + unK);
		
		// Construct grid belief "MDP" (over all unknown grid points_)
		mainLog.println("Building belief space approximation...");
		List<List<HashMap<Integer, Double>>> observationProbs = new ArrayList<>();//memoization for reuse
		List<List<HashMap<Integer, Belief>>> nextBelieves = new ArrayList<>();//memoization for reuse
		buildBeliefMDP(simplePOMDP, unknownGridPoints, observationProbs, nextBelieves);
		
		// HashMap for storing real time values for the discretized grid belief states
		HashMap<Belief, Double> vhash = new HashMap<>();
		HashMap<Belief, Double> vhash_backUp = new HashMap<>();
		for (Belief g : gridPoints) {
			if (unknownGridPoints.contains(g)) {
				vhash.put(g, 0.0);
				vhash_backUp.put(g, 0.0);
			} else {
				vhash.put(g, 1.0);
				vhash_backUp.put(g, 1.0);
			}
		}

		// Start iterations
		mainLog.println("Solving belief space approximation...");
		long timer2 = System.currentTimeMillis();
		double value, chosenValue;
		int iters = 0;
		boolean done = false;
		while (!done && iters < maxIters) {
			// Iterate over all (unknown) grid points
			for (int i = 0; i < unK; i++) {
				Belief b = unknownGridPoints.get(i);
				int numChoices = simplePOMDP.getNumChoicesForObservation(b.so);

				chosenValue = min ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
				for (int a = 0; a < numChoices; a++) {
					value = 0;
					for (Map.Entry<Integer, Double> entry : observationProbs.get(i).get(a).entrySet()) {
						int o = entry.getKey();
						double observationProb = entry.getValue();
						Belief nextBelief = nextBelieves.get(i).get(a).get(o);
						// find discretized grid points to approximate the nextBelief
						value += observationProb * interpolateOverGrid(o, nextBelief, vhash_backUp);
					}
					if ((min && chosenValue - value > 1.0e-6) || (!min && value - chosenValue > 1.0e-6)) {
						chosenValue = value;
					}
				}
				//update V(b) to the chosenValue
				vhash.put(b, chosenValue);
			}
			// Check termination
			done = PrismUtils.hashMapsAreClose(vhash, vhash_backUp, termCritParam, termCrit == TermCrit.RELATIVE);
			// back up	
			Set<Map.Entry<Belief, Double>> entries = vhash.entrySet();
			for (Map.Entry<Belief, Double> entry : entries) {
				vhash_backUp.put(entry.getKey(), entry.getValue());
			}
			iters++;
		}
		// Non-convergence is an error (usually)
		if (!done && errorOnNonConverge) {
			String msg = "Iterative method did not converge within " + iters + " iterations.";
			msg += "\nConsider using a different numerical method or increasing the maximum number of iterations";
			throw new PrismException(msg);
		}
		timer2 = System.currentTimeMillis() - timer2;
		mainLog.print("Belief space value iteration (" + (min ? "min" : "max") + ")");
		mainLog.println(" took " + iters + " iterations and " + timer2 / 1000.0 + " seconds.");

		// find discretized grid points to approximate the initialBelief
		Belief initialBelief = simplePOMDP.getInitialBelief();
		double outerBound = interpolateOverGrid(initialBelief.so, initialBelief, vhash_backUp);
		mainLog.println("Outer bound: " + outerBound);
		
		// Build DTMC to get inner bound (and strategy)
		mainLog.println("\nBuilding strategy-induced model...");
		List<Belief> listBeliefs = new ArrayList<>();
		MDPSimple mdp = buildStrategyModel(simplePOMDP, null, vhash, vhash_backUp, target, min, listBeliefs);
		mainLog.print("Strategy-induced model: " + mdp.infoString());
		// Export?
		if (stratFilename != null) {
			mdp.exportToPrismExplicitTra(stratFilename);
			mdp.exportToDotFile(stratFilename + ".dot", mdp.getLabelStates("target"));
//			for (int ii = 0; ii < mdp.getNumStates(); ii++) {
//				System.out.println(ii + ":" + listBeliefs.get(ii));
//			}
		}
		// Create MDP model checker (disable strat generation - if enabled, we want the POMDP one) 
		MDPModelChecker mcMDP = new MDPModelChecker(this);
		mcMDP.setExportAdv(false);
		mcMDP.setGenStrat(false);
		// Solve MDP to get inner bound
		ModelCheckerResult mcRes = mcMDP.computeReachProbs(mdp, mdp.getLabelStates("target"), true);
		double innerBound = mcRes.soln[0];
		mainLog.println("Inner bound: " + innerBound);

		// Finished fixed-resolution grid approximation
		timer = System.currentTimeMillis() - timer;
		mainLog.print("\nFixed-resolution grid approximation (" + (min ? "min" : "max") + ")");
		mainLog.println(" took " + timer / 1000.0 + " seconds.");

		// Extract Store result
		double lowerBound = Math.min(innerBound, outerBound);
		double upperBound = Math.max(innerBound, outerBound);
		mainLog.println("Result bounds: [" + lowerBound + "," + upperBound + "]");
		double soln[] = new double[simplePOMDP.getNumStates()];
		for (int initialState : simplePOMDP.getInitialStates()) {
			soln[initialState] = (lowerBound + upperBound) / 2.0;
		}

		// Return results
		ModelCheckerResult res = new ModelCheckerResult();
		res.soln = soln;
		res.precision = (upperBound - lowerBound) / 2.0;
		res.numIters = iters;
		res.timeTaken = timer / 1000.0;
		return res;
	}

	/**
	 * Compute expected reachability rewards.
	 * i.e. compute the min/max reward accumulated to reach a state in {@code target}.
	 * @param pomdp The POMDP
	 * @param mdpRewards The rewards
	 * @param target Target states
	 * @param min Min or max rewards (true=min, false=max)
	 */
	public ModelCheckerResult computeReachRewards(POMDP pomdp, MDPRewards mdpRewards, BitSet target, boolean min) throws PrismException
	{
		ModelCheckerResult res = null;
		long timer;
		String stratFilename = null;
		
		// Check for multiple initial states 
		if (pomdp.getNumInitialStates() > 1) {
			throw new PrismNotSupportedException("POMDP model checking does not yet support multiple initial states");
		}
		
		// Start expected reachability
		timer = System.currentTimeMillis();
		mainLog.println("\nStarting expected reachability (" + (min ? "min" : "max") + ")...");

		// If required, create/initialise strategy storage
		if (genStrat || exportAdv) {
			stratFilename = exportAdvFilename;
		}

		// Compute rewards
		res = computeReachRewardsFixedGrid(pomdp, mdpRewards, target, min, stratFilename);

		// Finished expected reachability
		timer = System.currentTimeMillis() - timer;
		mainLog.println("Expected reachability took " + timer / 1000.0 + " seconds.");

		// Update time taken
		res.timeTaken = timer / 1000.0;
		return res;
	}

	/**
	 * Compute expected reachability rewards using Lovejoy's fixed-resolution grid approach.
	 * Optionally, store optimal (memoryless) strategy info. 
	 * @param pomdp The POMMDP
	 * @param mdpRewards The rewards
	 * @param target Target states
	 * @param inf States for which reward is infinite
	 * @param min Min or max rewards (true=min, false=max)
	 * @param strat Storage for (memoryless) strategy choice indices (ignored if null)
	 */
	protected ModelCheckerResult computeReachRewardsFixedGrid(POMDP pomdp, MDPRewards mdpRewards, BitSet target, boolean min, String stratFilename) throws PrismException
	{
		if (!(pomdp instanceof POMDPSimple)) {
			throw new PrismException("Sorry, FixedGrid does not support POMDP other than POMDPSimple.");
		}
		POMDPSimple simplePOMDP = (POMDPSimple) pomdp;

		// Start fixed-resolution grid approximation
		long timer = System.currentTimeMillis();
		mainLog.println("Starting fixed-resolution grid approximation (" + (min ? "min" : "max") + ")...");

		// Find out the observations for the target states
		TreeSet<Integer> targetObservsSet = new TreeSet<>();
		for (int bit = target.nextSetBit(0); bit >= 0; bit = target.nextSetBit(bit + 1)) {
			targetObservsSet.add(simplePOMDP.getObservation(bit));
		}
		LinkedList<Integer> targetObservs = new LinkedList<>(targetObservsSet);

		// Initialise the grid points
		ArrayList<Belief> gridPoints = new ArrayList<>();//the set of grid points (discretized believes)
		ArrayList<Belief> unknownGridPoints = new ArrayList<>();//the set of unknown grid points (discretized believes)
		initialiseGridPoints(simplePOMDP, targetObservs, gridPoints, unknownGridPoints);
		int unK = unknownGridPoints.size();
		mainLog.print("Grid statistics: resolution=" + gridResolution);
		mainLog.println(", points=" + gridPoints.size() + ", unknown points=" + unK);
		
		// Construct grid belief "MDP" (over all unknown grid points_)
		mainLog.println("Building belief space approximation...");
		List<List<HashMap<Integer, Double>>> observationProbs = new ArrayList<>();// memoization for reuse
		List<List<HashMap<Integer, Belief>>> nextBelieves = new ArrayList<>();// memoization for reuse
		buildBeliefMDP(simplePOMDP, unknownGridPoints, observationProbs, nextBelieves);
		// Rewards
		List<List<Double>> rewards = new ArrayList<>(); // memoization for reuse
		for (int i = 0; i < unK; i++) {
			Belief b = unknownGridPoints.get(i);
			int numChoices = simplePOMDP.getNumChoicesForObservation(b.so);
			List<Double> action_reward = new ArrayList<>();// for memoization
			for (int a = 0; a < numChoices; a++) {
				action_reward.add(simplePOMDP.getCostAfterAction(b, a, mdpRewards)); // c(a,b)
			}
			rewards.add(action_reward);
		}
		
		// HashMap for storing real time values for the discretized grid belief states
		HashMap<Belief, Double> vhash = new HashMap<>();
		HashMap<Belief, Double> vhash_backUp = new HashMap<>();
		for (Belief g : gridPoints) {
			vhash.put(g, 0.0);
			vhash_backUp.put(g, 0.0);
		}
		
		// Start iterations
		mainLog.println("Solving belief space approximation...");
		long timer2 = System.currentTimeMillis();
		double value, chosenValue;
		int iters = 0;
		boolean done = false;
		while (!done && iters < maxIters) {
			// Iterate over all (unknown) grid points
			for (int i = 0; i < unK; i++) {
				Belief b = unknownGridPoints.get(i);
				int numChoices = simplePOMDP.getNumChoicesForObservation(b.so);
				chosenValue = min ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
				for (int a = 0; a < numChoices; a++) {
					value = rewards.get(i).get(a);
					for (Map.Entry<Integer, Double> entry : observationProbs.get(i).get(a).entrySet()) {
						int o = entry.getKey();
						double observationProb = entry.getValue();
						Belief nextBelief = nextBelieves.get(i).get(a).get(o);
						// find discretized grid points to approximate the nextBelief
						value += observationProb * interpolateOverGrid(o, nextBelief, vhash_backUp);
					}
					if ((min && chosenValue - value > 1.0e-6) || (!min && value - chosenValue > 1.0e-6)) {
						chosenValue = value;
					}
				}
				//update V(b) to the chosenValue
				vhash.put(b, chosenValue);
			}
			// Check termination
			done = PrismUtils.hashMapsAreClose(vhash, vhash_backUp, termCritParam, termCrit == TermCrit.RELATIVE);
			// back up	
			Set<Map.Entry<Belief, Double>> entries = vhash.entrySet();
			for (Map.Entry<Belief, Double> entry : entries) {
				vhash_backUp.put(entry.getKey(), entry.getValue());
			}
			iters++;
		}
		// Non-convergence is an error (usually)
		if (!done && errorOnNonConverge) {
			String msg = "Iterative method did not converge within " + iters + " iterations.";
			msg += "\nConsider using a different numerical method or increasing the maximum number of iterations";
			throw new PrismException(msg);
		}
		timer2 = System.currentTimeMillis() - timer2;
		mainLog.print("Belief space value iteration (" + (min ? "min" : "max") + ")");
		mainLog.println(" took " + iters + " iterations and " + timer2 / 1000.0 + " seconds.");

		// find discretized grid points to approximate the initialBelief
		Belief initialBelief = simplePOMDP.getInitialBelief();
		double outerBound = interpolateOverGrid(initialBelief.so, initialBelief, vhash_backUp);
		mainLog.println("Outer bound: " + outerBound);
		
		// Build DTMC to get inner bound (and strategy)
		mainLog.println("\nBuilding strategy-induced model...");
		List<Belief> listBeliefs = new ArrayList<>();
		MDPSimple mdp = buildStrategyModel(simplePOMDP, mdpRewards, vhash, vhash_backUp, target, min, listBeliefs);
		mainLog.print("Strategy-induced model: " + mdp.infoString());
		// Build rewards too
		MDPRewardsSimple mdpRewardsNew = new MDPRewardsSimple(mdp.getNumStates());
		int numStates = mdp.getNumStates();
		for (int ii = 0; ii < numStates; ii++) {
			if (mdp.getNumChoices(ii) > 0) {
				int action = ((Integer) mdp.getAction(ii, 0));
				double rew = simplePOMDP.getCostAfterAction(listBeliefs.get(ii), action, mdpRewards);
				mdpRewardsNew.addToStateReward(ii, rew);
			}
		}
		// Export?
		if (stratFilename != null) {
			mdp.exportToPrismExplicitTra(stratFilename);
			mdp.exportToDotFile(stratFilename + ".dot", mdp.getLabelStates("target"));
//			for (int ii = 0; ii < mdp.getNumStates(); ii++) {
//				System.out.println(ii + ":" + listBeliefs.get(ii));
//			}
		}

		// Create MDP model checker (disable strat generation - if enabled, we want the POMDP one) 
		MDPModelChecker mcMDP = new MDPModelChecker(this);
		mcMDP.setExportAdv(false);
		mcMDP.setGenStrat(false);
		// Solve MDP to get inner bound
		ModelCheckerResult mcRes = mcMDP.computeReachRewards(mdp, mdpRewardsNew, mdp.getLabelStates("target"), true);
		double innerBound = mcRes.soln[0];
		mainLog.println("Inner bound: " + innerBound);

		// Finished fixed-resolution grid approximation
		timer = System.currentTimeMillis() - timer;
		mainLog.print("\nFixed-resolution grid approximation (" + (min ? "min" : "max") + ")");
		mainLog.println(" took " + timer / 1000.0 + " seconds.");

		// Extract Store result
		double lowerBound = Math.min(innerBound, outerBound);
		double upperBound = Math.max(innerBound, outerBound);
		mainLog.println("Result bounds: [" + lowerBound + "," + upperBound + "]");
		double soln[] = new double[simplePOMDP.getNumStates()];
		for (int initialState : simplePOMDP.getInitialStates()) {
			soln[initialState] = (lowerBound + upperBound) / 2.0;
		}

		// Return results
		ModelCheckerResult res = new ModelCheckerResult();
		res.soln = soln;
		res.precision = (upperBound - lowerBound) / 2.0;
		res.numIters = iters;
		res.timeTaken = timer / 1000.0;
		return res;
	}

	/**
	 * Compute expected reachability rewards.
	 * i.e. compute the min/max reward accumulated to reach a state in {@code target}.
	 * @param pomdp The POMDP
	 * @param mdpRewards The rewards
	 * @param target Target states
	 * @param min Min or max rewards (true=min, false=max)
	 */
	public ModelCheckerResult computeMultiReachRewards(POMDP pomdp, MDPRewards mdpRewards1, MDPRewards mdpRewards2, BitSet target, boolean min, int numPoints) throws PrismException
	{
		ModelCheckerResult res = null;
		long timer;
		String stratFilename = null;
		
		// Check for multiple initial states 
		if (pomdp.getNumInitialStates() > 1) {
			throw new PrismNotSupportedException("POMDP model checking does not yet support multiple initial states");
		}
		
		// Start expected reachability
		timer = System.currentTimeMillis();
		mainLog.println("\nStarting expected reachability (" + (min ? "min" : "max") + ")...");

		// If required, create/initialise strategy storage
		if (genStrat || exportAdv) {
			stratFilename = exportAdvFilename;
		}

		// Compute rewards
		res = computeMultiReachRewardsFixedGrid(pomdp, mdpRewards1, mdpRewards2, target, min, numPoints, stratFilename);

		// Finished expected reachability
		timer = System.currentTimeMillis() - timer;
		mainLog.println("Expected reachability took " + timer / 1000.0 + " seconds.");

		// Update time taken
		res.timeTaken = timer / 1000.0;
		return res;
	}

	/**
	 * Compute expected reachability rewards using Lovejoy's fixed-resolution grid approach.
	 * Optionally, store optimal (memoryless) strategy info. 
	 * @param pomdp The POMMDP
	 * @param mdpRewards The rewards
	 * @param target Target states
	 * @param inf States for which reward is infinite
	 * @param min Min or max rewards (true=min, false=max)
	 * @param strat Storage for (memoryless) strategy choice indices (ignored if null)
	 */
	protected ModelCheckerResult computeMultiReachRewardsFixedGrid(POMDP pomdp, MDPRewards mdpRewards1, MDPRewards mdpRewards2, BitSet target, boolean min, int numPoints, String stratFilename) throws PrismException
	{
		if (!(pomdp instanceof POMDPSimple)) {
			throw new PrismException("Sorry, FixedGrid does not support POMDP other than POMDPSimple.");
		}
		POMDPSimple simplePOMDP = (POMDPSimple) pomdp;

		// Start fixed-resolution grid approximation
		long timer = System.currentTimeMillis();
		mainLog.println("Starting fixed-resolution grid approximation (" + (min ? "min" : "max") + ")...");

		// Find out the observations for the target states
		TreeSet<Integer> targetObservsSet = new TreeSet<>();
		for (int bit = target.nextSetBit(0); bit >= 0; bit = target.nextSetBit(bit + 1)) {
			targetObservsSet.add(simplePOMDP.getObservation(bit));
		}
		LinkedList<Integer> targetObservs = new LinkedList<>(targetObservsSet);

		// Initialise the grid points
		ArrayList<Belief> gridPoints = new ArrayList<>();//the set of grid points (discretized believes)
		ArrayList<Belief> unknownGridPoints = new ArrayList<>();//the set of unknown grid points (discretized believes)
		initialiseGridPoints(simplePOMDP, targetObservs, gridPoints, unknownGridPoints);
		int unK = unknownGridPoints.size();
		mainLog.print("Grid statistics: resolution=" + gridResolution);
		mainLog.println(", points=" + gridPoints.size() + ", unknown points=" + unK);
		
		// Construct grid belief "MDP" (over all unknown grid points_)
		mainLog.println("Building belief space approximation...");
		List<List<HashMap<Integer, Double>>> observationProbs = new ArrayList<>();// memoization for reuse
		List<List<HashMap<Integer, Belief>>> nextBelieves = new ArrayList<>();// memoization for reuse
		buildBeliefMDP(simplePOMDP, unknownGridPoints, observationProbs, nextBelieves);
		
		double innerBound = 0, outerBound = 0;
		int iters = 0;
		
		HashMap<Double,Pair<Double,Double>> paretoPoints = new HashMap<>();
		for (int w = 0; w < numPoints; w++) {

			double weights[] = new double[] { w / (numPoints - 1.0), 1.0 - w / (numPoints - 1.0) };
			mainLog.println("\nWeights: " + Arrays.toString(weights));

			// Rewards
			MDPRewardsSimple mdpRewards = new MDPRewardsSimple(simplePOMDP.getNumStates());
			for (int i = 0; i < simplePOMDP.getNumStates(); i++) {
				mdpRewards.setStateReward(i, weights[0] * mdpRewards1.getStateReward(i) + weights[1] * mdpRewards2.getStateReward(i));
				int numChoices = simplePOMDP.getNumChoices(i);
				for (int j = 0; j < numChoices; j++) {
					mdpRewards.setTransitionReward(i, j,
							weights[0] * mdpRewards1.getTransitionReward(i, j) + weights[1] * mdpRewards2.getTransitionReward(i, j));
				}
			}
			List<List<Double>> rewards = new ArrayList<>(); // memoization for reuse
			for (int i = 0; i < unK; i++) {
				Belief b = unknownGridPoints.get(i);
				int numChoices = simplePOMDP.getNumChoicesForObservation(b.so);
				List<Double> action_reward = new ArrayList<>();// for memoization
				for (int a = 0; a < numChoices; a++) {
					action_reward.add(simplePOMDP.getCostAfterAction(b, a, mdpRewards)); // c(a,b)
				}
				rewards.add(action_reward);
			}

			// HashMap for storing real time values for the discretized grid belief states
			HashMap<Belief, Double> vhash = new HashMap<>();
			HashMap<Belief, Double> vhash_backUp = new HashMap<>();
			for (Belief g : gridPoints) {
				vhash.put(g, 0.0);
				vhash_backUp.put(g, 0.0);
			}

			// Start iterations
			mainLog.println("Solving belief space approximation...");
			long timer2 = System.currentTimeMillis();
			double value, chosenValue;
			iters = 0;
			boolean done = false;
			while (!done && iters < maxIters) {
				// Iterate over all (unknown) grid points
				for (int i = 0; i < unK; i++) {
					Belief b = unknownGridPoints.get(i);
					int numChoices = simplePOMDP.getNumChoicesForObservation(b.so);
					chosenValue = min ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
					for (int a = 0; a < numChoices; a++) {
						value = rewards.get(i).get(a);
						for (Map.Entry<Integer, Double> entry : observationProbs.get(i).get(a).entrySet()) {
							int o = entry.getKey();
							double observationProb = entry.getValue();
							Belief nextBelief = nextBelieves.get(i).get(a).get(o);
							// find discretized grid points to approximate the nextBelief
							value += observationProb * interpolateOverGrid(o, nextBelief, vhash_backUp);
						}
						if ((min && chosenValue - value > 1.0e-6) || (!min && value - chosenValue > 1.0e-6)) {
							chosenValue = value;
						}
					}
					//update V(b) to the chosenValue
					vhash.put(b, chosenValue);
				}
				// Check termination
				done = PrismUtils.hashMapsAreClose(vhash, vhash_backUp, termCritParam, termCrit == TermCrit.RELATIVE);
				// back up	
				Set<Map.Entry<Belief, Double>> entries = vhash.entrySet();
				for (Map.Entry<Belief, Double> entry : entries) {
					vhash_backUp.put(entry.getKey(), entry.getValue());
				}
				iters++;
			}
			// Non-convergence is an error (usually)
			if (!done && errorOnNonConverge) {
				String msg = "Iterative method did not converge within " + iters + " iterations.";
				msg += "\nConsider using a different numerical method or increasing the maximum number of iterations";
				throw new PrismException(msg);
			}
			timer2 = System.currentTimeMillis() - timer2;
			mainLog.print("Belief space value iteration (" + (min ? "min" : "max") + ")");
			mainLog.println(" took " + iters + " iterations and " + timer2 / 1000.0 + " seconds.");

			// find discretized grid points to approximate the initialBelief
			Belief initialBelief = simplePOMDP.getInitialBelief();
			outerBound = interpolateOverGrid(initialBelief.so, initialBelief, vhash_backUp);
			mainLog.println("Outer bound: " + outerBound);

			// Build DTMC to get inner bound (and strategy)
			mainLog.println("\nBuilding strategy-induced model...");
			List<Belief> listBeliefs = new ArrayList<>();
			MDPSimple mdp = buildStrategyModel(simplePOMDP, mdpRewards, vhash, vhash_backUp, target, min, listBeliefs);
			mainLog.print("Strategy-induced model: " + mdp.infoString());
			// Build rewards too
			MDPRewardsSimple mdpRewardsNew = new MDPRewardsSimple(mdp.getNumStates());
			MDPRewardsSimple mdpRewardsNew1 = new MDPRewardsSimple(mdp.getNumStates());
			MDPRewardsSimple mdpRewardsNew2 = new MDPRewardsSimple(mdp.getNumStates());
			int numStates = mdp.getNumStates();
			for (int ii = 0; ii < numStates; ii++) {
				if (mdp.getNumChoices(ii) > 0) {
					int action = ((Integer) mdp.getAction(ii, 0));
					double rew1 = simplePOMDP.getCostAfterAction(listBeliefs.get(ii), action, mdpRewards1);
					double rew2 = simplePOMDP.getCostAfterAction(listBeliefs.get(ii), action, mdpRewards2);
					mdpRewardsNew.addToStateReward(ii, weights[0] * rew1 + weights[1] * rew2);
					mdpRewardsNew1.addToStateReward(ii, rew1);
					mdpRewardsNew2.addToStateReward(ii, rew2);
				}
			}
			// Export?
			if (stratFilename != null) {
				mdp.exportToPrismExplicitTra(stratFilename);
				mdp.exportToDotFile(stratFilename + ".dot", mdp.getLabelStates("target"));
				//			for (int ii = 0; ii < mdp.getNumStates(); ii++) {
				//				System.out.println(ii + ":" + listBeliefs.get(ii));
				//			}
			}

			// Create MDP model checker (disable strat generation - if enabled, we want the POMDP one) 
			MDPModelChecker mcMDP = new MDPModelChecker(this);
			mcMDP.setExportAdv(false);
			mcMDP.setGenStrat(false);
			// Solve MDP to get inner bound
			ModelCheckerResult mcRes = mcMDP.computeReachRewards(mdp, mdpRewardsNew, mdp.getLabelStates("target"), true);
			ModelCheckerResult mcRes1 = mcMDP.computeReachRewards(mdp, mdpRewardsNew1, mdp.getLabelStates("target"), true);
			ModelCheckerResult mcRes2 = mcMDP.computeReachRewards(mdp, mdpRewardsNew2, mdp.getLabelStates("target"), true);
			innerBound = mcRes.soln[0];
			mainLog.println("Inner bound: " + innerBound);
			double lowerBound = Math.min(innerBound, outerBound);
			double upperBound = Math.max(innerBound, outerBound);
			mainLog.println("Result bounds: [" + lowerBound + "," + upperBound + "]");
			mainLog.println("Individual objectives (for weights " + Arrays.toString(weights) + "): " + mcRes1.soln[0] + ", " + mcRes2.soln[0]);
			paretoPoints.put(weights[0], new Pair<Double,Double>(mcRes1.soln[0], mcRes2.soln[0]));
		}
		
		// Sort/uniquify point list and print
		mainLog.println("\nPareto point list: " + paretoPoints);
		mainLog.println("Pareto curve:");
		ArrayList<Pair<Double,Double>> paretoList = new ArrayList<>();
		for (Pair<Double,Double> point : paretoPoints.values()) {
			if (!paretoList.contains(point)) {
				paretoList.add(point);
				mainLog.println(point.first + "\t" + point.second);
			}
		}
		Comparator<Pair<Double,Double>> cmp = (Pair<Double,Double> o1, Pair<Double,Double> o2) -> (((Double) o1.first).compareTo(((Double) o2.first))); 
		Collections.sort(paretoList, cmp);
		//mainLog.println("Pareto: " + paretoList);
		
		// Finished fixed-resolution grid approximation
		timer = System.currentTimeMillis() - timer;
		mainLog.print("\nFixed-resolution grid approximation (" + (min ? "min" : "max") + ")");
		mainLog.println(" took " + timer / 1000.0 + " seconds.");

		// Extract Store result
		double lowerBound = Math.min(innerBound, outerBound);
		double upperBound = Math.max(innerBound, outerBound);
		mainLog.println("Result bounds: [" + lowerBound + "," + upperBound + "]");
		double soln[] = new double[simplePOMDP.getNumStates()];
		for (int initialState : simplePOMDP.getInitialStates()) {
			soln[initialState] = (lowerBound + upperBound) / 2.0;
		}

		// Return results
		ModelCheckerResult res = new ModelCheckerResult();
		res.soln = soln;
		res.precision = (upperBound - lowerBound) / 2.0;
		res.numIters = iters;
		res.timeTaken = timer / 1000.0;
		return res;
	}

	protected void initialiseGridPoints(POMDPSimple simplePOMDP, LinkedList<Integer> targetObservs, ArrayList<Belief> gridPoints, ArrayList<Belief> unknownGridPoints)
	{
		ArrayList<ArrayList<Double>> assignment;
		boolean isTargetObserv;
		int numObservations = simplePOMDP.getNumObservations();
		int numUnobservations = simplePOMDP.getNumUnobservations();
		int numStates = simplePOMDP.getNumStates();
		for (int so = 0; so < numObservations; so++) {
			ArrayList<Integer> unobservsForObserv = new ArrayList<>();
			for (int s = 0; s < numStates; s++) {
				if (so == simplePOMDP.getObservation(s)) {
					unobservsForObserv.add(simplePOMDP.getUnobservation(s));
				}
			}
			assignment = fullAssignment(unobservsForObserv.size(), gridResolution);

			isTargetObserv = targetObservs.isEmpty() ? false : ((Integer) targetObservs.peekFirst() == so);
			if (isTargetObserv) {
				targetObservs.removeFirst();
			}

			for (ArrayList<Double> inner : assignment) {
				double[] bu = new double[numUnobservations];
				int k = 0;
				for (int unobservForObserv : unobservsForObserv) {
					bu[unobservForObserv] = inner.get(k);
					k++;
				}

				Belief g = new Belief(so, bu);
				gridPoints.add(g);
				if (!isTargetObserv) {
					unknownGridPoints.add(g);
				}
			}
		}
	}
	
	protected void buildBeliefMDP(POMDPSimple simplePOMDP, ArrayList<Belief> unknownGridPoints, List<List<HashMap<Integer, Double>>> observationProbs, List<List<HashMap<Integer, Belief>>> nextBelieves)
	{
		int unK = unknownGridPoints.size();
		for (int i = 0; i < unK; i++) {
			Belief b = unknownGridPoints.get(i);
			double[] beliefInDist = b.toDistributionOverStates(simplePOMDP);
			//mainLog.println("Belief " + i + ": " + b);
			//mainLog.print("Belief dist:");
			//mainLog.println(beliefInDist);
			List<HashMap<Integer, Double>> action_observation_probs = new ArrayList<>();// for memoization
			List<HashMap<Integer, Belief>> action_observation_Believes = new ArrayList<>();// for memoization
			int numChoices = simplePOMDP.getNumChoicesForObservation(b.so);
			for (int a = 0; a < numChoices; a++) {
				//mainLog.println(i+"/"+unK+", "+a+"/"+numChoices);
				HashMap<Integer, Double> observation_probs = new HashMap<>();// for memoization
				HashMap<Integer, Belief> observation_believes = new HashMap<>();// for memoization
				simplePOMDP.computeObservationProbsAfterAction(beliefInDist, a, observation_probs);
				for (Map.Entry<Integer, Double> entry : observation_probs.entrySet()) {
					int o = entry.getKey();
					//mainLog.println(i+"/"+unK+", "+a+"/"+numChoices+", "+o+"/"+numObservations);
					Belief nextBelief = simplePOMDP.getBeliefAfterActionAndObservation(b, a, o);
					//mainLog.print(i + "/" + unK + ", " + a + "/" + numChoices + ", " + o + "/" + numObservations);
					//mainLog.println(" - " + entry.getValue() + ":" + nextBelief);
					observation_believes.put(o, nextBelief);
				}
				action_observation_probs.add(observation_probs);
				action_observation_Believes.add(observation_believes);
			}
			observationProbs.add(action_observation_probs);
			nextBelieves.add(action_observation_Believes);
		}
	}
	
	protected double interpolateOverGrid(int o, Belief belief, HashMap<Belief, Double> vhash)
	{
		ArrayList<double[]> subSimplex = new ArrayList<>();
		double[] lambdas = new double[belief.bu.length];
		getSubSimplexAndLambdas(belief.bu, subSimplex, lambdas, gridResolution);
		//calculate the approximate value for the belief
		double val = 0;
		for (int j = 0; j < lambdas.length; j++) {
			if (lambdas[j] >= 1e-6) {
				val += lambdas[j] * vhash.get(new Belief(o, subSimplex.get(j)));
			}
		}
		return val;
	}
	
	protected MDPSimple buildStrategyModel(POMDPSimple simplePOMDP, MDPRewards mdpRewards, HashMap<Belief, Double> vhash, HashMap<Belief, Double> vhash_backUp, BitSet target, boolean min, List<Belief> listBeliefs)
	{
		
		// extract optimal policy and store it in file named stratFilename
		Belief initialBelief = simplePOMDP.getInitialBelief();
		MDPSimple mdp = new MDPSimple();
		BitSet mdpTarget = new BitSet();
		IndexedSet<Belief> exploredBelieves = new IndexedSet<>(true);
		LinkedList<Belief> toBeExploredBelives = new LinkedList<>();
		exploredBelieves.add(initialBelief);
		toBeExploredBelives.offer(initialBelief);
		mdp.addState();
		mdp.addInitialState(0);
		int src = -1;
		while (!toBeExploredBelives.isEmpty()) {
			Belief b = toBeExploredBelives.pollFirst();
			src++;
			if (isTargetBelief(b.toDistributionOverStates(simplePOMDP), target)) {
				mdpTarget.set(src);
			}
			extractBestActions(src, b, vhash, simplePOMDP, mdpRewards, min, exploredBelieves, toBeExploredBelives, target, mdp);
		}
		
		mdp.addLabel("target", mdpTarget);
		listBeliefs.addAll(exploredBelieves.toArrayList());
		return mdp;
	}
	
	protected ArrayList<ArrayList<Integer>> assignGPrime(int startIndex, int min, int max, int length)
	{
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (startIndex == length - 1) {
			for (int i = min; i <= max; i++) {
				ArrayList<Integer> innerList = new ArrayList<>();
				innerList.add(i);
				result.add(innerList);
			}
		} else {
			for (int i = min; i <= max; i++) {
				ArrayList<ArrayList<Integer>> nextResult = assignGPrime(startIndex + 1, 0, i, length);
				for (ArrayList<Integer> nextReulstInner : nextResult) {
					ArrayList<Integer> innerList = new ArrayList<>();
					innerList.add(i);
					for (Integer a : nextReulstInner) {
						innerList.add(a);
					}
					result.add(innerList);
				}
			}
		}

		return result;
	}

	private ArrayList<ArrayList<Double>> fullAssignment(int length, int resolution)
	{
		ArrayList<ArrayList<Integer>> GPrime = assignGPrime(0, resolution, resolution, length);
		ArrayList<ArrayList<Double>> result = new ArrayList<ArrayList<Double>>();
		for (ArrayList<Integer> GPrimeInner : GPrime) {
			ArrayList<Double> innerList = new ArrayList<>();
			int i;
			for (i = 0; i < length - 1; i++) {
				int temp = GPrimeInner.get(i) - GPrimeInner.get(i + 1);
				innerList.add((double) temp / resolution);
			}
			innerList.add((double) GPrimeInner.get(i) / resolution);
			result.add(innerList);
		}
		return result;
	}

	private int[] getSortedPermutation(double[] inputArray)
	{
		int n = inputArray.length;
		double[] inputCopy = new double[n];
		int[] permutation = new int[n];
		int iState = 0, iIteration = 0;
		int iNonZeroEntry = 0, iZeroEntry = n - 1;
		boolean bDone = false;

		for (iState = n - 1; iState >= 0; iState--) {
			if (inputArray[iState] == 0.0) {
				inputCopy[iZeroEntry] = 0.0;
				permutation[iZeroEntry] = iState;
				iZeroEntry--;
			}

		}

		for (iState = 0; iState < n; iState++) {
			if (inputArray[iState] != 0.0) {
				inputCopy[iNonZeroEntry] = inputArray[iState];
				permutation[iNonZeroEntry] = iState;
				iNonZeroEntry++;
			}
		}

		while (!bDone) {
			bDone = true;
			for (iState = 0; iState < iNonZeroEntry - iIteration - 1; iState++) {
				if (inputCopy[iState] < inputCopy[iState + 1]) {
					swap(inputCopy, iState, iState + 1);
					swap(permutation, iState, iState + 1);
					bDone = false;
				}
			}
			iIteration++;
		}

		return permutation;
	}

	private void swap(int[] aiArray, int i, int j)
	{
		int temp = aiArray[i];
		aiArray[i] = aiArray[j];
		aiArray[j] = temp;
	}

	private void swap(double[] aiArray, int i, int j)
	{
		double temp = aiArray[i];
		aiArray[i] = aiArray[j];
		aiArray[j] = temp;
	}

	protected boolean getSubSimplexAndLambdas(double[] b, ArrayList<double[]> subSimplex, double[] lambdas, int resolution)
	{
		int n = b.length;
		int M = resolution;

		double[] X = new double[n];
		int[] V = new int[n];
		double[] D = new double[n];
		for (int i = 0; i < n; i++) {
			X[i] = 0;
			for (int j = i; j < n; j++) {
				X[i] += M * b[j];
			}
			X[i] = Math.round(X[i] * 1e6) / 1e6;
			V[i] = (int) Math.floor(X[i]);
			D[i] = X[i] - V[i];
		}

		int[] P = getSortedPermutation(D);
		//		mainLog.println("X: "+ Arrays.toString(X));
		//		mainLog.println("V: "+ Arrays.toString(V));
		//		mainLog.println("D: "+ Arrays.toString(D));
		//		mainLog.println("P: "+ Arrays.toString(P));

		ArrayList<int[]> Qs = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			int[] Q = new int[n];
			if (i == 0) {
				for (int j = 0; j < n; j++) {
					Q[j] = V[j];
				}
				Qs.add(Q);
			} else {
				for (int j = 0; j < n; j++) {
					if (j == P[i - 1]) {
						Q[j] = Qs.get(i - 1)[j] + 1;
					} else {
						Q[j] = Qs.get(i - 1)[j];
					}

				}
				Qs.add(Q);
			}
			//			mainLog.println(Arrays.toString(Q));
		}

		for (int[] Q : Qs) {
			double[] node = new double[n];
			int i;
			for (i = 0; i < n - 1; i++) {
				int temp = Q[i] - Q[i + 1];
				node[i] = (double) temp / M;
			}
			node[i] = (double) Q[i] / M;
			subSimplex.add(node);
		}

		double sum = 0;
		for (int i = 1; i < n; i++) {
			double lambda = D[P[i - 1]] - D[P[i]];
			lambdas[i] = lambda;
			sum = sum + lambda;
		}
		lambdas[0] = 1 - sum;

		for (int i = 0; i < n; i++) {
			double sum2 = 0;
			for (int j = 0; j < n; j++) {
				sum2 += lambdas[j] * subSimplex.get(j)[i];
			}
			//			mainLog.println("b["+i+"]: "+b[i]+"  b^[i]:"+sum2);
			if (Math.abs(b[i] - sum2) > 1e-4) {
				return false;
			}

		}
		return true;
	}

	/**
	 * Find the best action for this belief state, add the belief state to the list
	 * of ones examined so far, and store the strategy info. We store this as an MDP.
	 * @param belief Belief state to examine
	 * @param vhash
	 * @param simplePOMDP
	 * @param mdpRewards
	 * @param min
	 * @param beliefList
	 */
	protected void extractBestActions(int src, Belief belief, HashMap<Belief, Double> vhash, POMDPSimple simplePOMDP, MDPRewards mdpRewards, boolean min,
			IndexedSet<Belief> exploredBelieves, LinkedList<Belief> toBeExploredBelives, BitSet target, MDPSimple mdp)
	{
		if (isTargetBelief(belief.toDistributionOverStates(simplePOMDP), target)) {
			// Add self-loop
			/*Distribution distr = new Distribution();
			distr.set(src, 1);
			mdp.addActionLabelledChoice(src, distr, null);*/
			return;
		}
			
		double[] beliefInDist = belief.toDistributionOverStates(simplePOMDP);
		double chosenValue = min ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
		int chosenActionIndex = -1;
		ArrayList<Integer> bestActions = new ArrayList<>();
		List<Double> action_reward = new ArrayList<>();
		List<HashMap<Integer, Double>> action_observation_probs = new ArrayList<>();
		List<HashMap<Integer, Belief>> action_observation_Believes = new ArrayList<>();
		//evaluate each action in b
		int numChoices = simplePOMDP.getNumChoicesForObservation(belief.so);
		for (int a = 0; a < numChoices; a++) {
			double value = 0;
			if (mdpRewards != null) {
				value = simplePOMDP.getCostAfterAction(belief, a, mdpRewards); // c(a,b)	
			}
			// Build/store successor observations, probabilities and resulting beliefs
			HashMap<Integer, Double> observation_probs = new HashMap<>();
			HashMap<Integer, Belief> observation_believes = new HashMap<>();
			simplePOMDP.computeObservationProbsAfterAction(beliefInDist, a, observation_probs);
			for (Map.Entry<Integer, Double> entry : observation_probs.entrySet()) {
				int o = entry.getKey();
				Belief nextBelief = simplePOMDP.getBeliefAfterActionAndObservation(belief, a, o);
				observation_believes.put(o, nextBelief);
				double observationProb = observation_probs.get(o);
				value += observationProb * interpolateOverGrid(o, nextBelief, vhash);
			}
			// Store the list of observations, probabilities and resulting beliefs for this action
			action_observation_probs.add(observation_probs);
			action_observation_Believes.add(observation_believes);

			//select action that minimizes/maximizes Q(a,b), i.e. value
			if ((min && chosenValue - value > 1.0e-6) || (!min && value - chosenValue > 1.0e-6))//value<bestValue
			{
				chosenValue = value;
				chosenActionIndex = a;
				bestActions.clear();
				bestActions.add(chosenActionIndex);
			} else if (Math.abs(value - chosenValue) < 1.0e-6)//value==chosenValue
			{
				//random tie broker
				chosenActionIndex = Math.random() < 0.5 ? a : chosenActionIndex;
				bestActions.clear();
				bestActions.add(a);
			}
		}

		Distribution distr = new Distribution();
		for (Integer a : bestActions) {
			for (Map.Entry<Integer, Double> entry : action_observation_probs.get(a).entrySet()) {
				int o = entry.getKey();
				double observationProb = entry.getValue();
				Belief nextBelief = action_observation_Believes.get(a).get(o);
				if (exploredBelieves.add(nextBelief)) {
					// If so, add to the explore list
					toBeExploredBelives.add(nextBelief);
					// And to model
					mdp.addState();
				}
				// Get index of state in state set
				int dest = exploredBelieves.getIndexOfLastAdd();
				distr.add(dest, observationProb);
			}
		}
		// Add transition distribution, with choice _index_ encoded as action
		mdp.addActionLabelledChoice(src, distr, bestActions.get(0));
	}
	
	public static boolean isTargetBelief(double[] belief, BitSet target)
	{
		 double prob=0;
		 for (int i = target.nextSetBit(0); i >= 0; i = target.nextSetBit(i+1)) 
		 {
			 prob+=belief[i];
		 }
		 if(Math.abs(prob-1.0)<1.0e-6)
		 {
			 return true;
		 }
		 return false;
	}	

	/**
	 * Simple test program.
	 */
	public static void main(String args[])
	{
		POMDPModelChecker mc;
		POMDPSimple pomdp;
		ModelCheckerResult res;
		BitSet init, target;
		Map<String, BitSet> labels;
		boolean min = true;
		try {
			mc = new POMDPModelChecker(null);
			MDPSimple mdp = new MDPSimple();
			mdp.buildFromPrismExplicit(args[0]);
			//mainLog.println(mdp);
			labels = mc.loadLabelsFile(args[1]);
			//mainLog.println(labels);
			init = labels.get("init");
			target = labels.get(args[2]);
			if (target == null)
				throw new PrismException("Unknown label \"" + args[2] + "\"");
			for (int i = 3; i < args.length; i++) {
				if (args[i].equals("-min"))
					min = true;
				else if (args[i].equals("-max"))
					min = false;
				else if (args[i].equals("-nopre"))
					mc.setPrecomp(false);
			}
			pomdp = new POMDPSimple(mdp);
			res = mc.computeReachRewards(pomdp, null, target, min);
			System.out.println(res.soln[init.nextSetBit(0)]);
		} catch (PrismException e) {
			System.out.println(e);
		}
	}
}