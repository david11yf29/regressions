const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

const { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower"],
  labelColumns: ["mpg"]
});

// console.log(features, labels);

const regression = new LinearRegression(features, labels, {
  learningRate: 0.001, iterations: 1
});

regression.train();

console.log('Updated M is: ', regression.m, 'Updated B is: ', regression.b);
