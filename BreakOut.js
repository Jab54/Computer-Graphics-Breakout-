//Wesam Jaber

const shaderSource = `
   struct Uniforms {
      color: vec4f,
      matrix: mat4x4f,
    };

    @group(0) @binding(0) var<uniform> uni: Uniforms;

    struct VSOutput {
        @builtin(position) position : vec4f,
    };
    @vertex
    fn vertexMain(@location(0) position: vec2f) -> VSOutput {
        var vsOut: VSOutput;
        vsOut.position = uni.matrix * vec4f(position, 0.0, 1.0);
        return vsOut;
    }

    @fragment
    fn fragmentMain(vsOut: VSOutput) -> @location(0) vec4f 
    {
        return uni.color;
    }
`;

let unitSqrVerts = new Float32Array([
    -0.5,-0.5,
     0.5,-0.5,
     0.5, 0.5,
    -0.5,-0.5,
     0.5, 0.5,
    -0.5, 0.5,
]);

// World coordinates
let bottom = -5.0;
let upper  =  5.0;
let left   = -10.0;
let right  =  10.0;

// block size
let wid = 3.5;
let hgh = 1.0;

// Game objects
// Ball speed configuration
const BALL_SPEED = 0.04; // lower = slower
const ball = {x1 : -0.5, x2 : 0.5, y1 : -0.5, y2 : 0.5, dX : BALL_SPEED, dY : BALL_SPEED};
// Ensure ball always has some vertical component after collisions
const MIN_VERTICAL_RATIO = 0.25; // fraction of speed that must be vertical
const paddle = {x1 : -0.5 * wid, x2 : 0.5 * wid, y1 : -0.5 * hgh, y2 : 0.5 * hgh, speed : 0.1};

// Block constructor
function block() {
  this.x1 = -0.5 * wid;
  this.x2 = 0.5 * wid;
  this.y1 = -0.5 * hgh;
  this.y2 = 0.5 * hgh;
  this.vis = true;
}

// WebGPU objects
let device, context, pipeline, shader;
let canvasFormat;
let vertexBuffers = [];
let nVertsPerShape = [];

const IntanceData = [];
let nInstances = 0;

// Game state
let blocks = []; // array of block objects
let nBlocksRow = 5;
let nBlocksCol = 4;
let visBlocks = true;
let strikes = 0;
let maxStrikes = 3;
let running = true;

// input
let leftPressed = false;
let rightPressed = false;

async function initWebGPU() {
  if (!navigator.gpu) throw Error('WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();
  const canvas = document.getElementById('webgpuCanvas');
  const aspect = canvas.width / canvas.height;
  left   = bottom * aspect;
  right  = upper * aspect;
  context = canvas.getContext('webgpu');
  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: canvasFormat, alphaMode: 'premultiplied' });
  device.pushErrorScope('validation');
  shader = device.createShaderModule({ code: shaderSource });
  const err = await device.popErrorScope();
  if (err) throw Error('Shader compilation error');
}

function pipelineSetUp() {
  const vertexBufferLayout = [{ arrayStride: 2*4, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] }];
  pipeline = device.createRenderPipeline({
    label: 'breakout-pipe',
    layout: 'auto',
    vertex: { module: shader, entryPoint: 'vertexMain', buffers: vertexBufferLayout },
    fragment: { module: shader, entryPoint: 'fragmentMain', targets: [{ format: canvasFormat }] },
    primitive: { topology: 'triangle-list' }
  });
}

function geomSetUp() {
  const sqrVBO = device.createBuffer({ size: unitSqrVerts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(sqrVBO, 0, unitSqrVerts);
  vertexBuffers.push(sqrVBO);
  const bytesPerVert = 8;
  const nVerts = unitSqrVerts.byteLength / bytesPerVert;
  nVertsPerShape.push(nVerts);

  // Create instances for blocks + ball + paddle
  // blocks first
  for (let r = 0; r < nBlocksCol; ++r) {
    for (let c = 0; c < nBlocksRow; ++c) {
      blocks.push(new block());
    }
  }

  // Instances count = blocks + ball + paddle
  nInstances = blocks.length + 2;

  for (let i = 0; i < nInstances; ++i) {
    const uniformBufferSize = (4 + 16) * 4;
    const uniformBuffer = device.createBuffer({ size: uniformBufferSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const uniformValues = new Float32Array(uniformBufferSize / 4);
    const kColorOffset = 0;
    const kMatrixOffset = 4;
    const colorValue = uniformValues.subarray(kColorOffset, kColorOffset + 4);
    // assign colors: blocks green, ball red, paddle blue
    if (i < blocks.length) colorValue.set([0.2, 0.8, 0.2, 1]);
    else if (i === blocks.length) colorValue.set([1, 0.2, 0.2, 1]);
    else colorValue.set([0.2, 0.4, 1, 1]);

    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uniformBuffer } }] });
    const xForm = { translation: [0,0,0], rotation: [0,0,0], scale: [1,1,1] };
    IntanceData.push({ uniformBuffer, uniformValues, matrixValue: uniformValues.subarray(kMatrixOffset, kMatrixOffset+16), bindGroup, xForm });
  }

  // Position blocks in grid near top
  const startY = 3.5;
  const gapX = 0.2;
  const gapY = 0.15;
  let idx = 0;
  for (let r = 0; r < nBlocksCol; ++r) {
    for (let c = 0; c < nBlocksRow; ++c) {
      const x = (c - (nBlocksRow-1)/2) * (wid + gapX);
      const y = startY - r * (hgh + gapY);
      IntanceData[idx].xForm.translation = [x, y, 0];
      IntanceData[idx].xForm.scale = [wid, hgh, 1];
      idx++;
    }
  }

  // Ball instance (index = blocks.length)
  const ballIndex = blocks.length;
  IntanceData[ballIndex].xForm.scale = [1.0, 1.0, 1.0];
  IntanceData[ballIndex].xForm.translation = [0.0, 0.0, 0.0];

  // Paddle instance (index = blocks.length + 1)
  const paddleIndex = blocks.length + 1;
  IntanceData[paddleIndex].xForm.scale = [wid, hgh, 1.0];
  IntanceData[paddleIndex].xForm.translation = [0.0, -4.0, 0.0];
}

function simulate() {
  if (!visBlocks || !running) return;
  const ballIndex = blocks.length;
  const paddleIndex = blocks.length + 1;

  // Move ball
  IntanceData[ballIndex].xForm.translation[0] += ball.dX;
  IntanceData[ballIndex].xForm.translation[1] += ball.dY;

  // Move paddle
  if (rightPressed) IntanceData[paddleIndex].xForm.translation[0] += paddle.speed;
  if (leftPressed) IntanceData[paddleIndex].xForm.translation[0] -= paddle.speed;

  // Clamp paddle to canvas
  const px = IntanceData[paddleIndex].xForm.translation[0];
  const halfPaddle = wid * 0.5;
  if (px - halfPaddle < left) IntanceData[paddleIndex].xForm.translation[0] = left + halfPaddle;
  if (px + halfPaddle > right) IntanceData[paddleIndex].xForm.translation[0] = right - halfPaddle;

  ballPaddleIntersect(ballIndex, paddleIndex);
  collideWithCanvas(ballIndex);

  // Test collision with blocks
  for (let i = 0; i < blocks.length; ++i) {
    if (!blocks[i].vis) continue;
    if (ballBlockXsect(i)) {
      blocks[i].vis = false;
      // mark instance invisible by scaling to zero (simple approach)
      IntanceData[i].xForm.scale = [0.0001, 0.0001, 1];
      break;
    }
  }

  // Check win
  const anyVisible = blocks.some(b => b.vis);
  if (!anyVisible) {
    running = false;
    document.getElementById('message').textContent = 'You Win!';
    showReplay(true);
  }
}

function showReplay(visible) {
  const btn = document.getElementById('replay');
  if (!btn) return;
  btn.style.display = visible ? 'inline-block' : 'none';
}

function resetGame() {
  // Reset game state
  strikes = 0;
  running = true;
  // Reset blocks
  for (let i = 0; i < blocks.length; ++i) {
    blocks[i].vis = true;
    IntanceData[i].xForm.scale = [wid, hgh, 1];
  }
  // Reset ball and paddle positions
  const ballIndex = blocks.length;
  IntanceData[ballIndex].xForm.translation = [0, 0, 0];
  IntanceData[blocks.length+1].xForm.translation = [0, -4.0, 0];
  // Reset velocities
  ball.dX = BALL_SPEED * (Math.random() > 0.5 ? 1 : -1);
  ball.dY = BALL_SPEED;
  ensureMinimumVertical();
  // Clear message and hide replay
  const msg = document.getElementById('message');
  if (msg) msg.textContent = '';
  showReplay(false);
}

function ballPaddleIntersect(ballIndex, paddleIndex) {
  const bx = IntanceData[ballIndex].xForm.translation[0];
  const by = IntanceData[ballIndex].xForm.translation[1];
  const px = IntanceData[paddleIndex].xForm.translation[0];
  const py = IntanceData[paddleIndex].xForm.translation[1];

  const ballLeft = bx + ball.x1;
  const ballRight = bx + ball.x2;
  const ballTop = by + ball.y2;
  const ballBottom = by + ball.y1;

  const paddleLeft = px + paddle.x1;
  const paddleRight = px + paddle.x2;
  const paddleTop = py + paddle.y2;
  const paddleBottom = py + paddle.y1;

  if (ballRight >= paddleLeft && ballLeft <= paddleRight && ballBottom <= paddleTop && ballTop >= paddleBottom) {
    // compute current speed magnitude
    const speedMag = Math.hypot(ball.dX, ball.dY) || BALL_SPEED;

    // deflect based on where the ball hits the paddle
    const offset = (bx - px) / (wid * 0.5); // -1..1
  // bias lateral change so center hits go mostly up, edge hits add sideways velocity
  let newDX = ball.dX + offset * 0.5;
  // ensure paddle always sends the ball upward
  let newDY = Math.abs(ball.dY);

    // normalize and keep speed magnitude
    const len = Math.hypot(newDX, newDY);
    if (len > 1e-6) {
      ball.dX = (newDX / len) * speedMag;
      ball.dY = (newDY / len) * speedMag;
    }

    // Nudge ball just above the paddle to avoid repeated collisions
    IntanceData[ballIndex].xForm.translation[1] = paddleTop + (ball.y2 - ball.y1) * 0.5 + 0.01;
    // ensure there is a minimum vertical component so the ball doesn't go perfectly horizontal
    ensureMinimumVertical();
    return true;
  }
  return false;
}

function collideWithCanvas(ballIndex) {
  const bx = IntanceData[ballIndex].xForm.translation[0];
  const by = IntanceData[ballIndex].xForm.translation[1];
  const ballLeft = bx + ball.x1;
  const ballRight = bx + ball.x2;
  const ballTop = by + ball.y2;
  const ballBottom = by + ball.y1;

  if (ballLeft <= left) { ball.dX = Math.abs(ball.dX); }
  if (ballRight >= right) { ball.dX = -Math.abs(ball.dX); }
  if (ballTop >= upper) { ball.dY = -Math.abs(ball.dY); }

  // bottom -> strike
  if (ballBottom <= bottom) {
    strikes++;
    if (strikes >= maxStrikes) {
      running = false;
      document.getElementById('message').textContent = 'You Lose!';
      showReplay(true);
    } else {
      // reset ball and paddle
      IntanceData[ballIndex].xForm.translation = [0, 0, 0];
      IntanceData[blocks.length+1].xForm.translation[0] = 0;
  // Reset to configured BALL_SPEED
  ball.dX = BALL_SPEED * (Math.random() > 0.5 ? 1 : -1);
  ball.dY = BALL_SPEED;
      ensureMinimumVertical();
    }
  }
}

function ballBlockXsect(i) {
  const ballIndex = blocks.length;
  const bx = IntanceData[ballIndex].xForm.translation[0];
  const by = IntanceData[ballIndex].xForm.translation[1];
  const bLeft = bx + ball.x1;
  const bRight = bx + ball.x2;
  const bTop = by + ball.y2;
  const bBottom = by + ball.y1;

  const ix = IntanceData[i].xForm.translation[0];
  const iy = IntanceData[i].xForm.translation[1];
  const iLeft = ix + blocks[i].x1 * IntanceData[i].xForm.scale[0];
  const iRight = ix + blocks[i].x2 * IntanceData[i].xForm.scale[0];
  const iTop = iy + blocks[i].y2 * IntanceData[i].xForm.scale[1];
  const iBottom = iy + blocks[i].y1 * IntanceData[i].xForm.scale[1];

  const overlapX = Math.min(bRight, iRight) - Math.max(bLeft, iLeft);
  const overlapY = Math.min(bTop, iTop) - Math.max(bBottom, iBottom);

  if (overlapX > 0 && overlapY > 0) {
    // determine shallowest penetration direction to decide reflection axis
    const speedMag = Math.hypot(ball.dX, ball.dY) || BALL_SPEED;
    if (overlapX < overlapY) {
      // horizontal collision -> reflect X
      ball.dX = -ball.dX;
      // nudge ball outside block horizontally
      if (bx < ix) {
        IntanceData[ballIndex].xForm.translation[0] = iLeft - ball.x2 - 0.01;
      } else {
        IntanceData[ballIndex].xForm.translation[0] = iRight - ball.x1 + 0.01;
      }
    } else {
      // vertical collision -> reflect Y
      ball.dY = -ball.dY;
      // nudge ball outside block vertically
      if (by < iy) {
        // ball hit block from below
        IntanceData[ballIndex].xForm.translation[1] = iBottom - ball.y2 - 0.01;
      } else {
        // ball hit block from above
        IntanceData[ballIndex].xForm.translation[1] = iTop - ball.y1 + 0.01;
      }
    }

    // keep overall speed magnitude (avoid slowdowns or speedups)
    const len = Math.hypot(ball.dX, ball.dY);
    if (len > 1e-6) {
      ball.dX = (ball.dX / len) * speedMag;
      ball.dY = (ball.dY / len) * speedMag;
    }

    // ensure vertical component after block bounce
    ensureMinimumVertical();

    return true;
  }
  return false;
}

// Ensure ball's vertical component is at least a fraction of total speed
function ensureMinimumVertical() {
  const speed = Math.hypot(ball.dX, ball.dY) || BALL_SPEED;
  const minV = speed * MIN_VERTICAL_RATIO;
  if (Math.abs(ball.dY) < minV) {
    // preserve horizontal sign, prefer upward when currently zero
    const signY = ball.dY === 0 ? 1 : Math.sign(ball.dY); // prefer upward when zero (towards top)
    ball.dY = signY * minV;
    // adjust dX so speed magnitude is preserved
    const newDX = Math.sign(ball.dX) * Math.sqrt(Math.max(0, speed*speed - ball.dY*ball.dY));
    ball.dX = newDX;
  }
}

function render() {
  const bgColor = [0.8,0.8,0.8];
  const c = document.getElementById('webgpuCanvas');
  const sel = document.getElementById('bgselect');
  if (sel) {
    const v = Number(sel.value);
    const colors = [[1,0.8,0.8],[0.7,0.9,0.7],[0.8,0.8,1],[0.8,0.8,0.8],[0,0,0]];
    bgColor[0] = colors[v][0]; bgColor[1] = colors[v][1]; bgColor[2] = colors[v][2];
  }

  simulate();

  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', clearValue: { r: bgColor[0], g: bgColor[1], b: bgColor[2], a: 1 }, storeOp: 'store' }] });
  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vertexBuffers[0]);

  // draw instances
  for (let i = 0; i < IntanceData.length; ++i) {
    const { uniformBuffer, uniformValues, matrixValue, bindGroup, xForm } = IntanceData[i];

    // start with ortho
    const adjustedBottom = bottom;
    const adjustedUpper = upper;
    const adjustedLeft = left;
    const adjustedRight = right;
    mat4.ortho(adjustedLeft, adjustedRight, adjustedBottom, adjustedUpper, -400, 400, matrixValue);

    // apply object transforms
    mat4.translate(matrixValue, xForm.translation, matrixValue);
    mat4.rotateX(matrixValue, xForm.rotation[0], matrixValue);
    mat4.rotateY(matrixValue, xForm.rotation[1], matrixValue);
    mat4.rotateZ(matrixValue, xForm.rotation[2], matrixValue);
    mat4.scale(matrixValue, xForm.scale, matrixValue);

    device.queue.writeBuffer(uniformBuffer, 0, uniformValues);
    pass.setBindGroup(0, bindGroup);
    pass.draw(nVertsPerShape[0]);
  }

  pass.end();
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
}

function setupInput() {
  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') {
      leftPressed = true;
      e.preventDefault();
    }
    if (e.key === 'ArrowRight') {
      rightPressed = true;
      e.preventDefault();
    }
    if (e.key === ' ') {
      running = !running;
      e.preventDefault();
    }
  });
  window.addEventListener('keyup', (e) => {
    if (e.key === 'ArrowLeft') {
      leftPressed = false;
      e.preventDefault();
    }
    if (e.key === 'ArrowRight') {
      rightPressed = false;
      e.preventDefault();
    }
  });
}

// Hook up replay button after DOM is ready
window.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('replay');
  if (btn) btn.addEventListener('click', resetGame);
});

async function init() {
  try {
    await initWebGPU();
    pipelineSetUp();
  } catch (e) {
    document.getElementById('message').textContent = 'WebGPU init failed: '+e;
    return;
  }
  geomSetUp();
  setupInput();
  document.getElementById('bgselect').value = '4';
  requestAnimationFrame(render);
}

window.onload = init;
