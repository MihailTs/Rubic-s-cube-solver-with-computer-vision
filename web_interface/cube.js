let s1 = (sketch) => {
    let rotX = 0;
    let rotY = 0;
    let cubeSide;

    sketch.setup = function() {
        let cnv = sketch.createCanvas(2 * sketch.windowWidth / 5, 3 * sketch.windowHeight / 4, sketch.WEBGL);
        cnv.parent('cubeContainer');
        cubeSide = sketch.width / 2.6;
        let camZ = 2.5 * (sketch.height / 2) / sketch.tan(sketch.PI / 6);
        sketch.perspective(sketch.PI / 6, sketch.width / sketch.height, 0.1, 10000);
        sketch.camera(0, 0, camZ, 0, 0, 0, 0, 1, 0);
        sketch.frameRate(15);
    }

    sketch.draw = function() {
        sketch.background(200);
        drawRubiksCube(0, 0, 0, cubeSide);
    }

    function drawFace(faceName, size, gap, faceSide) {
        sketch.push();
        
        let half = size * 1.5;
        let sticker = size;
        
        // Position and rotate based on which face
        switch(faceName) {
            case 'front':
                sketch.translate(0, 0, half);
                break;
            case 'back':
                sketch.translate(0, 0, -half);
                sketch.rotateY(sketch.PI);
                break;
            case 'right':
                sketch.translate(half, 0, 0);
                sketch.rotateY(sketch.HALF_PI);
                break;
            case 'left':
                sketch.translate(-half, 0, 0);
                sketch.rotateY(-sketch.HALF_PI);
                break;
            case 'top':
                sketch.translate(0, -half, 0);
                sketch.rotateX(sketch.HALF_PI);
                break;
            case 'bottom':
                sketch.translate(0, half, 0);
                sketch.rotateX(-sketch.HALF_PI);
                break;
        }
        
        let index = 0;
        for (let row = -1; row <= 1; row++) {
            for (let col = -1; col <= 1; col++) {
                sketch.push();
                sketch.translate(col * (sticker + gap), row * (sticker + gap), 0);

                let colorKey = faceSide[index];
                let rgb = sharedCubeState.color_dict[colorKey];
                sketch.fill(rgb[0], rgb[1], rgb[2]);
                sketch.stroke(0);
                sketch.beginShape();
                sketch.vertex(-sticker/2, -sticker/2, 0);
                sketch.vertex(sticker/2, -sticker/2, 0);
                sketch.vertex(sticker/2, sticker/2, 0);
                sketch.vertex(-sticker/2, sticker/2, 0);
                sketch.endShape(sketch.CLOSE);
                
                index++;
                sketch.pop();
            }
        }
        
        sketch.pop();
    }

    // Then use it like this in drawRubiksCube:
    function drawRubiksCube(centerX, centerY, centerZ, size) {
        sketch.push();
        sketch.translate(centerX, centerY, centerZ);
        sketch.rotateX(rotX);
        sketch.rotateY(rotY);
        
        let small = size / 3;
        let gap = 1;
        
        // Draw all 6 faces
        drawFace('front', small, gap, sharedCubeState.frontSide);
        drawFace('back', small, gap, sharedCubeState.backSide);
        drawFace('left', small, gap, sharedCubeState.leftSide);
        drawFace('right', small, gap, sharedCubeState.rightSide);
        drawFace('top', small, gap, sharedCubeState.topSide);
        drawFace('bottom', small, gap, sharedCubeState.bottomSide);
        
        sketch.pop();
    }

    sketch.mouseDragged = function() {
        rotY += sketch.movedX * 0.005;
        rotX -= sketch.movedY * 0.005;
    }
};

new p5(s1);