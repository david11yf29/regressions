const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

const { features, labels, testFeatures, testLabels } = loadCSV('./CarPrice_Assignment.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["curbweight", "horsepower", "highwaympg"],
  labelColumns: ["pricee"]
});

console.log(features, labels);

const regression = new LinearRegression(features, labels, {
  learningRate: 0.02, iterations: 40, batchSize: 10
});

regression.train();

plot({ 
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
})


console.log("b", regression.weights.arraySync()[0]);
console.log("x1", regression.weights.arraySync()[1]);
console.log("x2", regression.weights.arraySync()[2]);
console.log("x3", regression.weights.arraySync()[3]);
// console.log("MSE History", regression.mseHistory);

// console.log('Updated M is: ', regression.weights.arraySync()[1], 'Updated B is: ', regression.weights.arraySync()[0]);

// regression.predict([
//   [3505,182,22]
// ]).print();


