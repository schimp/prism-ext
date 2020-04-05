//==============================================================================
//	
//	Copyright (c) 2020-
//	Authors:
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

package prism;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import param.BigRational;
import param.ParamResult;
import parser.Values;
import parser.ast.ConstantList;
import parser.ast.Expression;
import parser.type.Type;
import parser.type.TypeBool;
import parser.type.TypeDouble;
import parser.type.TypeInt;
import parser.type.TypeVoid;

public class ResultTesting
{
	/**
	 * Tests a result (specified as a Result object) against the expected result,
	 * given by a string extracted from a RESULT: specification.
	 * If the test fails or something else goes wrong, an explanatory PrismException is thrown.
	 * Otherwise, the method successfully exits, returning a boolean value that indicates
	 * whether or not a check was actually applied (i.e. if the result specification is of the
	 * form "RESULT: ?") then false is returned; otherwise true.   
	 * @param strExpected Expected result
	 * @param constValues The values of any undefined constants (null or "" if none)
	 * @param type The type of the property yielding this result
	 * @param result The result to check
	 * @return Whether or not the check was performed
	 */
	public static boolean checkAgainstExpectedResultString(String strExpected, Values constValues, Type type, Result resultObj) throws PrismException
	{
		Object result = resultObj.getResult();
		// Check for special "don't care" case
		if (strExpected.equals("?")) {
			return false;
		}
		// Check for errors/exceptions
		if (strExpected.startsWith("Error") && !(result instanceof Exception)) {
			throw new PrismException("Was expecting an error");
		}
		if (result instanceof Exception) {
			return checkExceptionAgainstExpectedResultString(strExpected, (Exception) result);
		}
		// Check result of parametric model checking
		if (result instanceof ParamResult) {
			return ((ParamResult) result).test(type, strExpected, constValues);
		}
		// Otherwise, check depends on the type of the property  
		// Boolean-valued properties
		if (type instanceof TypeBool) {
			checkBooleanAgainstExpectedResultString(strExpected, constValues, resultObj);
		}
		// Integer-valued properties (non-exact mode)
		else if (type instanceof TypeInt && !(result instanceof BigRational)) {
			checkIntAgainstExpectedResultString(strExpected, constValues, resultObj);
		}
		// Double-valued properties (non-exact mode)
		else if (type instanceof TypeDouble && !(result instanceof BigRational)) {
			checkDoubleAgainstExpectedResultString(strExpected, constValues, resultObj);
		}
		// Double-valued or integer-valued properties exact mode)
		else if ((type instanceof TypeDouble || type instanceof TypeInt) && result instanceof BigRational) {
			checkExactAgainstExpectedResultString(strExpected, constValues, type, resultObj);
		}
		else if (type instanceof TypeVoid && result instanceof TileList) { //Pareto curve
			checkParetoAgainstExpectedResultString(strExpected, constValues, resultObj);
		}
		// Unknown type
		else {
			throw new PrismException("Don't know how to test properties of type " + type);
		}
		return true;
	}

	/**
	 * Tests a result specified as an Exception object against the expected result,
	 * given by a string extracted from a RESULT: specification.
	 * If the test fails or something else goes wrong, an explanatory PrismException is thrown.
	 * Otherwise, the method successfully exits, returning a boolean value that indicates
	 * whether or not a check was actually applied.   
	 * @param strExpected Expected result
	 * @param result The actual result
	 * @return Whether or not the check was performed
	 */
	private static boolean checkExceptionAgainstExpectedResultString(String strExpected, Exception result) throws PrismException
	{
		String errMsg = result.getMessage();
		if (strExpected.startsWith("Error")) {
			// handle expected errors
			if (strExpected.startsWith("Error:")) {
				String words[] = strExpected.substring(6).split(",");
				for (String word : words) {
					if (word.length() == 0) {
						throw new PrismException("Invalid RESULT specification: no expected words immediately following 'Error:'");
					}
					if (!errMsg.toLowerCase().contains(word)) {
						throw new PrismException("Error message should contain \"" + word + "\"");
					}
				}
			}
			return true;
		}
		if (result instanceof PrismNotSupportedException) {
			// not supported -> handle in caller
			throw (PrismNotSupportedException)result;
		}
		throw new PrismException("Unexpected error: " + errMsg);
	}
	
	/**
	 * Tests a result of type boolean against the expected result,
	 * given by a string extracted from a RESULT: specification.
	 * If the test fails or something else goes wrong, an explanatory PrismException is thrown.
	 * @param strExpected Expected result
	 * @param result The actual result
	 */
	private static void checkBooleanAgainstExpectedResultString(String strExpected, Values constValues, Result resultObj) throws PrismException
	{
		// Extract actual result
		Object result = resultObj.getResult();
		if (!(result instanceof Boolean)) {
			throw new PrismException("Result is wrong type (" + result.getClass() + ") for (boolean-valued) property");
		}
		boolean boolRes = ((Boolean) result).booleanValue();
		// Parse expected result
		boolean boolExp;
		if (strExpected.toLowerCase().equals("true")) {
			boolExp = true;
		} else if (strExpected.toLowerCase().equals("false")) {
			boolExp = false;
		} else {
			boolExp = parseExpectedResultString(strExpected, constValues).evaluateBoolean(constValues);
			strExpected += " = " + boolExp; 	
		}
		// Check result
		if (boolRes != boolExp) {
			throw new PrismException("Wrong result (expected " + strExpected + ", got " + boolRes + ")");
		}
	}
	
	/**
	 * For the case where the expected result string is (or is suspected to be)
	 * a PRISM expression, perhaps in terms of constants, parse and return it.
	 * Throws an exception (invalid RESULT spec) if parsing fails.  
	 */
	private static Expression parseExpectedResultString(String strExpected, Values constValues) throws PrismException
	{
		try {
			Expression expectedExpr = Prism.parseSingleExpressionString(strExpected);
			expectedExpr = (Expression) expectedExpr.findAllConstants(new ConstantList(constValues));
			expectedExpr.typeCheck();
			return expectedExpr;
		} catch (PrismLangException e) {
			throw new PrismException("Invalid RESULT specification \"" + strExpected + "\": " + e.getMessage());
		}
	}
	
	/**
	 * Tests a result of type int against the expected result,
	 * given by a string extracted from a RESULT: specification.
	 * If the test fails or something else goes wrong, an explanatory PrismException is thrown.
	 * @param strExpected Expected result
	 * @param result The actual result
	 */
	private static void checkIntAgainstExpectedResultString(String strExpected, Values constValues, Result resultObj) throws PrismException
	{
		// Extract actual result
		Object result = resultObj.getResult();
		if (!(result instanceof Integer)) {
			throw new PrismException("Result is wrong type (" + result.getClass() + ") for (integer-valued) property");
		}
		int intRes = ((Integer) result).intValue();
		// Parse expected result
		int intExp;
		// See if it's an int literal
		try {
			intExp = Integer.parseInt(strExpected);
		}
		// If not, could be an expression
		catch (NumberFormatException e) {
			intExp = parseExpectedResultString(strExpected, constValues).evaluateInt(constValues);
			strExpected += " = " + intExp; 	
		}
		// Check result
		if (intRes != intExp) {
			throw new PrismException("Wrong result (expected " + strExpected + ", got " + intRes + ")");
		}
	}
	
	/**
	 * Tests a result of type double against the expected result,
	 * given by a string extracted from a RESULT: specification.
	 * If the test fails or something else goes wrong, an explanatory PrismException is thrown.
	 * @param strExpected Expected result
	 * @param result The actual result
	 */
	private static void checkDoubleAgainstExpectedResultString(String strExpected, Values constValues, Result resultObj) throws PrismException
	{
		// Extract actual result
		Object result = resultObj.getResult();
		if (!(result instanceof Double)) {
			throw new PrismException("Result is wrong type (" + result.getClass() + ") for (double-valued) property");
		}
		double doubleRes = ((Double) result).doubleValue();
		// Parse expected result
		double doubleExp;
		// First, handle ~... expected results
		// In this case (non-exact mode) it does not really matter,
		// as we currently do an approximate comparison anyway
		if (strExpected.startsWith("~")) {
			strExpected = strExpected.substring(1);
		}
		// See if it's NaN
		if (strExpected.equals("NaN")) {
			doubleExp = Double.NaN;
		}
		// See if it's a fraction
		else if (strExpected.matches("-?[0-9]+/[0-9]+")) {
			doubleExp = new BigRational(strExpected).doubleValue();
			strExpected += " = " + doubleExp; 	
		}
		// See if it's a double literal
		else try {
			doubleExp = Double.parseDouble(strExpected);
		}
		// If not, could be an expression
		catch (NumberFormatException e) {
			doubleExp = parseExpectedResultString(strExpected, constValues).evaluateDouble(constValues);
			strExpected += " = " + doubleExp; 	
		}
		// Check result
		if (Double.isNaN(doubleRes) || Double.isNaN(doubleExp)) {
			if (Double.isNaN(doubleRes) != Double.isNaN(doubleExp)) {
				throw new PrismException("Wrong result (expected " + strExpected + ", got " + doubleRes + ")");
			}	
		} else if (!PrismUtils.doublesAreCloseRel(doubleExp, doubleRes, 1e-5)) {
			throw new PrismException("Wrong result (expected " + strExpected + ", got " + doubleRes + ")");
		}
	}
	
	/**
	 * Tests an exact result (of type int or double) against the expected result,
	 * given by a string extracted from a RESULT: specification.
	 * If the test fails or something else goes wrong, an explanatory PrismException is thrown.
	 * @param strExpected Expected result
	 * @param result The actual result
	 */
	private static void checkExactAgainstExpectedResultString(String strExpected, Values constValues, Type type, Result resultObj) throws PrismException
	{
		// Extract actual result
		Object result = resultObj.getResult();
		if (!(result instanceof BigRational)) {
			throw new PrismException("Result is wrong type (" + result.getClass() + ") for exact property");
		}
		BigRational rationalRes = (BigRational) result;
		// Parse expected result
		BigRational rationalExp = null;
		// Deal with NaN separately
		if (strExpected.equals("NaN")) {
			if (rationalRes.isNaN()) {
				return;
			} else {
				throw new PrismException("Wrong result (expected NaN, got " + rationalRes + ")");
			}
		}
		// Make a note if we only need to do an approximate comparison
		// (i.e., a floating point string starting with ~)
		boolean approx = false; 
		if (strExpected.startsWith("~")) {
			approx = true;
			strExpected = strExpected.substring(1);
		}
		// See if it's an int/double/rational literal - parse with BigRational
		try {
			rationalExp = new BigRational(strExpected);
		}
		// If not, could be an expression
		catch (NumberFormatException e) {
			rationalExp = parseExpectedResultString(strExpected, constValues).evaluateExact(constValues);
		}
		// Check result
		if (!rationalRes.equals(rationalExp)) {
			boolean match = false;
			if (type instanceof TypeDouble) {
				// Try imprecise comparison
				try {
					double doubleExp = Double.parseDouble(strExpected);
					boolean areClose = PrismUtils.doublesAreCloseRel(doubleExp, rationalRes.doubleValue(), 1e-5);
					if (areClose) {
						if (approx) {
							// we only have an approximate value to compare to, so we are fine here
							match = true;
						} else {
							throw new PrismException("Inexact, but close result (expected '" + strExpected + "' = " + rationalExp + " ("
									+ rationalExp.toApproximateString() + "), got " + rationalRes + " (" + rationalRes.toApproximateString() + "))");
						}
					}
				} catch (NumberFormatException e) {
				}
			}
			if (!match) {
				throw new PrismException("Wrong result (expected '" + strExpected + "' = " + rationalExp + " (" + rationalExp.toApproximateString() + "), got "
						+ rationalRes + " (" + rationalRes.toApproximateString() + "))");
			}
		}
	}
	
	/**
	 * Tests a Pareto curve result against the expected result,
	 * given by a string extracted from a RESULT: specification.
	 * If the test fails or something else goes wrong, an explanatory PrismException is thrown.
	 * @param strExpected Expected result
	 * @param result The actual result
	 */
	private static void checkParetoAgainstExpectedResultString(String strExpected, Values constValues, Result resultObj) throws PrismException
	{
		// Create the list of points from the expected results
		List<Point> liExpected = new ArrayList<Point>();
		Pattern p = Pattern.compile("\\(([^,]*),([^)]*)\\)");
		Matcher m = p.matcher(strExpected);
		if (!m.find()) {
			throw new PrismException("The expected result does not contain any points, or does not have the required format.");
		}
		
		do {
			double x = Double.parseDouble(m.group(1));
			double y = Double.parseDouble(m.group(2));
			Point point = new Point(new double[] {x,y});
			liExpected.add(point);
		} while(m.find());

		Object result = resultObj.getResult();
		List<Point> liResult = ((TileList) result).getRealPoints();

		if (liResult.size() != liExpected.size()) {
			throw new PrismException("The expected Pareto curve and the computed Pareto curve have a different number of points.");
		}
		// Check if we can find a matching point for every point on the expected Pareto curve
		for(Point point : liExpected) {
			boolean foundClose = false;
			for(Point point2 : liResult) {
				if (point2.isCloseTo(point)) {
					foundClose = true;
					break;
				}
			}
			if (!foundClose) {
				throw new PrismException("The point " + point + " in the expected Pareto curve has no match among the points in the computed Pareto curve.");
			}
		}
		// Check if we can find a matching point for every point on the computed Pareto curve
		// (we did check if both lists have the same number of points, but that does
		// not rule out the possibility of two very similar points contained in one list)
		for (Point point : liResult) {
			boolean foundClose = false;
			for (Point point2 : liExpected) {
				if (point2.isCloseTo(point)) {
					foundClose = true;
					break;
				}
			}
			if (!foundClose) {
				throw new PrismException("The point " + point + " in the computed Pareto curve has no match among the points in the expected Pareto curve");
			}
		}
	}
}
