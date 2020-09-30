const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

const { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"]
});

// console.log(features, labels);

const regression = new LinearRegression(features, labels, {
  learningRate: 10, iterations: 100
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

console.log('R2 is', r2);

// console.log('Updated M is: ', regression.weights.arraySync()[1], 'Updated B is: ', regression.weights.arraySync()[0]);


