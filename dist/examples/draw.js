const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

let offsetX = 0;
let offsetY = 0;

function initCanvas(width, height) {
	document.body.appendChild(canvas);
	canvas.width = width;
	canvas.height = height
}

function offsetCanvas(x, y) {
	offsetX = x;
	offsetY = y;
}

function drawPoints(points) {
	// draw small square outline for each point
	const size = 6;
	ctx.strokeStyle = '#25baff';
	ctx.lineWidth = 2;

	for (const { x, y } of points) {
		ctx.strokeRect(x - size / 2, y - size / 2, size, size);
	}
}

function drawBezierCurves(curves) {
    if (curves.length < 1) return; // Early return if there are no curves to draw

    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1;

    let previousControlCCW, previousPoint, previousControlCW;

	ctx.beginPath();



	for(let i = 0; i < curves.length; i+= 4){
		const point = curves[i];
		const controlCCW = curves[i+1];
		const controlCW = curves[i+2];

        if (i === 0) {
			ctx.moveTo(point.x + offsetX, point.y + offsetY);
        } else {
            // For subsequent segments, use the previous segment's last control point and the current segment's control points and end point
            ctx.bezierCurveTo(
				previousControlCW.x + offsetX,
				previousControlCW.y + offsetY,
				controlCCW.x + offsetX,
				controlCCW.y + offsetY,
				point.x + offsetX,
				point.y + offsetY
			);
        }

		previousControlCCW = controlCCW;
		previousPoint = point;
		previousControlCW = controlCW;
    };

	// Close the path
	const firstPoint = curves[0]
	const firstControlCCW = curves[1];

	ctx.bezierCurveTo(
		previousControlCW.x + offsetX,
		previousControlCW.y + offsetY,
		firstControlCCW.x + offsetX,
		firstControlCCW.y + offsetY,
		firstPoint.x + offsetX,
		firstPoint.y + offsetY
	);

	ctx.stroke();
	ctx.closePath();
}

function drawControlPoints(curves) {
	ctx.lineWidth = 1;

	const pointSize = 4;
	const pointColor = 'white';
	const controlSize = 2;
	const controlColor = 'red';

	// draw circles for control points
	for(let i = 0; i < curves.length; i+= 4){
		const point = curves[i];
		const controlCCW = curves[i+1];
		const controlCW = curves[i+2];

		ctx.beginPath();
		ctx.strokeStyle = controlColor;
		ctx.arc(controlCCW.x + offsetX, controlCCW.y + offsetY, controlSize, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.closePath();

		ctx.beginPath();
		ctx.strokeStyle = controlColor;
		ctx.arc(controlCW.x + offsetX, controlCW.y + offsetY, controlSize, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.closePath();

		ctx.beginPath();
		ctx.strokeStyle = pointColor;
		ctx.arc(point.x + offsetX, point.y + offsetY, pointSize, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.closePath();
	};
}

function drawText(text, x, y, color = 'black', font = '12px sans-serif') {
	ctx.fillStyle = color;
	ctx.font = font;
	ctx.fillText(text, x, y);
}

function clearPaths() {
	ctx.clearRect(0, 0, canvas.width, canvas.height);
}
