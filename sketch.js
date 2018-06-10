// Matthew Panzer June 2018
// Polynomial Regression with TensorFlow.js

// Original code & concept: Daniel Shiffman
// http://codingtra.in
// http://patreon.com/codingtrain

// Linear Regression with TensorFlow.js
// Daniel Shiffman's Video: https://www.youtube.com/watch?v=dLp10CFIvxI

let x_vals = [];
let y_vals = [];

let a; // TF variable for storing coefficients
let N = 10; // Maximum function degree. Must be greater than 2

// When displaying the function: how many parts to break the
//  curve into. Higher number = better quality
const SEGMENTS = 100;
const POINT_SIZE = 24;
const LINE_WIDTH = 14;

curveX = [];

// DOM elements
let Slider;
let NumPointsP;
let ResetButton;

// ML parameters. Feel free to play around with these
const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);

// Helper to get the function coefficient at index n
const asub = n => a.slice(n, 1).squeeze();

function setup() {
  // Create a p5 canvas
  createCanvas(600, 600);
  colorMode(HSB);

  // Create our TF variable-- the coefficients
  a = tf.variable(tf.tensor1d(Array.from({ length: N }, () => random(1))));

  // Create our DOM Elements
  createP("Adjust slider to see various degrees");
  Slider = createSlider(1, N, 2, 1);
  NumPointsP = createP("2");
  ResetButton = createButton("Clear Points");
  ResetButton.mousePressed(clearPoints);

  // Create the X coordinates for drawing the curve.
  // This can be done here because they're the same every time.
  for (let i = 0; i <= SEGMENTS; i++) {
    curveX.push(map(i, 0, SEGMENTS, -1, 1));
  }
}

// Our loss function. Mean Squared Error
function loss(pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = ax^2 + bx + c ...
  let ns = Array.from({ length: N }).map((e, i) => i);
  const ys = ns.reduce(
    (acc, n) => acc.add(n === 0 ? asub(n) : xs.pow(tf.scalar(n)).mul(asub(n))),
    tf.ones([x.length])
  );
  return ys;
}

function mousePressed() {
  if (mouseX > width || mouseY > height) return;
  let x = map(mouseX, 0, width, -1, 1);
  let y = map(mouseY, 0, height, 1, -1);
  x_vals.push(x);
  y_vals.push(y);
}

function clearPoints() {
  x_vals = [];
  y_vals = [];
}

function draw() {
  // Update values based on Slider
  N = Slider.value();
  NumPointsP.html(N);

  // Optimize our coefficient values
  tf.tidy(() => {
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), ys));
    }
  });

  background(0);

  stroke(255);
  strokeWeight(POINT_SIZE);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);
    point(px, py);
  }

  // Make a curve prediction based on our model
  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();

  // Show the curve! (secretly a bunch of lines... shh!)
  noFill();
  strokeWeight(LINE_WIDTH);
  let prevX;
  let prevY;
  for (let j = 0; j <= SEGMENTS; j++) {
    const xPoint = map(curveX[j], -1, 1, 0, width);
    const yPoint = map(curveY[j], -1, 1, height, 0);
    let hue = map(j, 0, SEGMENTS, 0, 360);
    stroke(hue, 100, 100);

    if (prevX !== undefined && prevY !== undefined) {
      line(prevX, prevY, xPoint, yPoint);
    }

    prevX = xPoint;
    prevY = yPoint;
  }
}
