let s2 = (sketch) => {
    let squareSize;
    let padding;
    let sidePadding;
    let dropZones = [];

    sketch.setup = function () {
        let cnv = sketch.createCanvas(2 * sketch.windowWidth / 5, 3 * sketch.windowHeight / 4);
        cnv.parent('unfoldContainer');
        squareSize = sketch.width / 19;
        padding = squareSize / 5;
        sidePadding = 3 * squareSize / 2;
        sketch.frameRate(15);

        setupImageInputs(cnv.position().x, cnv.position().y);
        setupDragAndDrop();
    }

    sketch.draw = function () {
        sketch.background(200);
        addTextInstructions();
        drawSide(topCenterCoordinates(), sharedCubeState.topSide);
        drawSide(bottomCenterCoordinates(), sharedCubeState.bottomSide);
        drawSide(frontCenterCoordinates(), sharedCubeState.frontSide);
        drawSide(backCenterCoordinates(), sharedCubeState.backSide);
        drawSide(leftCenterCoordinates(), sharedCubeState.leftSide);
        drawSide(rightCenterCoordinates(), sharedCubeState.rightSide);
        
        // Highlight drop zones on hover during drag
        highlightDropZones();
    }

    function addTextInstructions() {
        sketch.textSize(15);
        sketch.fill(0, 0, 40); 
        sketch.text('Click or drag & drop an image on a center square to add its corresponding side image.', 30, 20);
        sketch.text('Click other squares to change color manually.', 180, 40);
    }

    function setupDragAndDrop() {
        // Store drop zone information
        dropZones = [
            { coords: topCenterCoordinates(), side: sharedCubeState.topSide, name: 'top' },
            { coords: frontCenterCoordinates(), side: sharedCubeState.frontSide, name: 'front' },
            { coords: leftCenterCoordinates(), side: sharedCubeState.leftSide, name: 'left' },
            { coords: rightCenterCoordinates(), side: sharedCubeState.rightSide, name: 'right' },
            { coords: bottomCenterCoordinates(), side: sharedCubeState.bottomSide, name: 'bottom' },
            { coords: backCenterCoordinates(), side: sharedCubeState.backSide, name: 'back' }
        ];

        // Add drag and drop event listeners to the canvas using p5's canvas element
        const canvas = sketch.canvas;
        
        canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
        });

        canvas.addEventListener('dragenter', (e) => {
            e.preventDefault();
        });

        canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();

            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                // Get mouse position relative to canvas
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                console.log('Drop at:', x, y);

                // Find which drop zone was hit
                const targetZone = findDropZone(x, y);
                if (targetZone) {
                    console.log('Dropped on zone:', targetZone.name);
                    handleSideImageInput(files[0], targetZone.side);
                } else {
                    console.log('No zone found at this location');
                }
            }
        });
    }

    function findDropZone(mouseX, mouseY) {
        for (let zone of dropZones) {
            const [centerX, centerY] = zone.coords;
            // Check if mouse is within the center square
            if (mouseX > centerX - squareSize / 2 && 
                mouseX < centerX + squareSize / 2 && 
                mouseY > centerY - squareSize / 2 && 
                mouseY < centerY + squareSize / 2) {
                return zone;
            }
        }
        return null;
    }

    function highlightDropZones() {
        // This could be enhanced to show visual feedback during drag
        // For now, it's a placeholder for future enhancement
    }

    function setupImageInputs(xoff, yoff) {
        createCenterFileInput(
            topCenterCoordinates()[0] + xoff,
            topCenterCoordinates()[1] + yoff,
            squareSize,
            img => handleSideImageInput(img, sharedCubeState.topSide)
        );
        createCenterFileInput(
            frontCenterCoordinates()[0] + xoff,
            frontCenterCoordinates()[1] + yoff,
            squareSize,
            img => handleSideImageInput(img, sharedCubeState.frontSide)
        );
        createCenterFileInput(
            leftCenterCoordinates()[0] + xoff,
            leftCenterCoordinates()[1] + yoff,
            squareSize,
            img => handleSideImageInput(img, sharedCubeState.leftSide)
        );
        createCenterFileInput(
            rightCenterCoordinates()[0] + xoff,
            rightCenterCoordinates()[1] + yoff,
            squareSize,
            img => handleSideImageInput(img, sharedCubeState.rightSide)
        );
        createCenterFileInput(
            bottomCenterCoordinates()[0] + xoff,
            bottomCenterCoordinates()[1] + yoff,
            squareSize,
            img => handleSideImageInput(img, sharedCubeState.bottomSide)
        );
        createCenterFileInput(
            backCenterCoordinates()[0] + xoff,
            backCenterCoordinates()[1] + yoff,
            squareSize,
            img => handleSideImageInput(img, sharedCubeState.backSide)
        );
    }

    function createCenterFileInput(x, y, size, onImageLoaded) {
        const input = sketch.createFileInput(file => {
            if (file.type !== "image") return;

            onImageLoaded(file.file);
        });

        input.position(x - size / 2, y - size / 2);
        input.size(size, size);

        // Make it invisible but clickable
        input.style("opacity", "0");
        input.style("cursor", "pointer");

        return input;
    }

    function drawSide(coordinates, sideColors) {
        let x = coordinates[0];
        let y = coordinates[1];
        sketch.fill(getColor(sideColors[4]));
        sketch.square(x - squareSize / 2, y - squareSize / 2, squareSize, 5);
        sketch.fill(getColor(sideColors[7]));
        sketch.square(x - squareSize / 2, y - squareSize / 2 + squareSize + padding, squareSize, 5);
        sketch.fill(getColor(sideColors[1]));
        sketch.square(x - squareSize / 2, y - squareSize / 2 - squareSize - padding, squareSize, 5);
        sketch.fill(getColor(sideColors[5]));
        sketch.square(x - squareSize / 2 + padding + squareSize, y - squareSize / 2, squareSize, 5);
        sketch.fill(getColor(sideColors[8]));
        sketch.square(x - squareSize / 2 + padding + squareSize, y - squareSize / 2 + squareSize + padding, squareSize, 5);
        sketch.fill(getColor(sideColors[2]));
        sketch.square(x - squareSize / 2 + padding + squareSize, y - squareSize / 2 - squareSize - padding, squareSize, 5);
        sketch.fill(getColor(sideColors[3]));
        sketch.square(x - squareSize / 2 - padding - squareSize, y - squareSize / 2, squareSize, 5);
        sketch.fill(getColor(sideColors[6]));
        sketch.square(x - squareSize / 2 - padding - squareSize, y - squareSize / 2 + squareSize + padding, squareSize, 5);
        sketch.fill(getColor(sideColors[0]));
        sketch.square(x - squareSize / 2 - padding - squareSize, y - squareSize / 2 - squareSize - padding, squareSize, 5);
    }

    function topCenterCoordinates() {
        return [(sketch.width / 2) - padding - (3 * squareSize / 2) - sidePadding / 2,
        sketch.height / 2 - (3 * squareSize) - (2 * padding) - sidePadding];
    }

    function bottomCenterCoordinates() {
        return [(sketch.width / 2) - padding - (3 * squareSize / 2) - sidePadding / 2,
        sketch.height / 2 + (3 * squareSize) + (2 * padding) + sidePadding];
    }

    function frontCenterCoordinates() {
        return [(sketch.width / 2) - padding - (3 * squareSize / 2) - sidePadding / 2,
        sketch.height / 2];
    }

    function backCenterCoordinates() {
        return [(sketch.width / 2) + (3 * padding) + (9 * squareSize / 2) + (3 * sidePadding / 2),
        sketch.height / 2];
    }

    function rightCenterCoordinates() {
        return [(sketch.width / 2) + padding + (3 * squareSize / 2) + sidePadding / 2,
        sketch.height / 2];
    }

    function leftCenterCoordinates() {
        return [(sketch.width / 2) - (3 * padding) - (9 * squareSize / 2) - (3 * sidePadding / 2),
        sketch.height / 2];
    }

    function getColor(a) {
        // console.log(sharedCubeState.color_dict)
        return sketch.color(
            sharedCubeState.color_dict[a][0],
            sharedCubeState.color_dict[a][1],
            sharedCubeState.color_dict[a][2]
        );
    }

    function checkSideClicked(centerCoordinates, checkSide) {
        // centers are not changable
        if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 && sketch.mouseX < centerCoordinates[0] + squareSize / 2 && sketch.mouseY > centerCoordinates[1] - squareSize / 2 - squareSize - padding && sketch.mouseY < centerCoordinates[1] - squareSize / 2 - padding) {
            checkSide[1] = (checkSide[1] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 && sketch.mouseX < centerCoordinates[0] + squareSize / 2 && sketch.mouseY > centerCoordinates[1] + squareSize / 2 + padding && sketch.mouseY < centerCoordinates[1] + squareSize / 2 + squareSize + padding) {
            checkSide[7] = (checkSide[7] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 - squareSize - padding && sketch.mouseX < centerCoordinates[0] - squareSize / 2 - padding && sketch.mouseY > centerCoordinates[1] - squareSize / 2 && sketch.mouseY < centerCoordinates[1] + squareSize / 2) {
            checkSide[3] = (checkSide[3] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] + squareSize / 2 + padding && sketch.mouseX < centerCoordinates[0] + squareSize / 2 + squareSize + padding && sketch.mouseY > centerCoordinates[1] - squareSize / 2 && sketch.mouseY < centerCoordinates[1] + squareSize / 2) {
            checkSide[5] = (checkSide[5] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 - squareSize - padding && sketch.mouseX < centerCoordinates[0] - squareSize / 2 - padding && sketch.mouseY > centerCoordinates[1] - squareSize / 2 - squareSize - padding && sketch.mouseY < centerCoordinates[1] - squareSize / 2 - padding) {
            checkSide[0] = (checkSide[0] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] + squareSize / 2 + padding && sketch.mouseX < centerCoordinates[0] + squareSize / 2 + squareSize + padding && sketch.mouseY > centerCoordinates[1] - squareSize / 2 - squareSize - padding && sketch.mouseY < centerCoordinates[1] - squareSize / 2 - padding) {
            checkSide[2] = (checkSide[2] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 - squareSize - padding && sketch.mouseX < centerCoordinates[0] - squareSize / 2 - padding && sketch.mouseY > centerCoordinates[1] + squareSize / 2 + padding && sketch.mouseY < centerCoordinates[1] + squareSize / 2 + squareSize + padding) {
            checkSide[6] = (checkSide[6] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] + squareSize / 2 + padding && sketch.mouseX < centerCoordinates[0] + squareSize / 2 + squareSize + padding && sketch.mouseY > centerCoordinates[1] + squareSize / 2 + padding && sketch.mouseY < centerCoordinates[1] + squareSize / 2 + squareSize + padding) {
            checkSide[8] = (checkSide[8] + 1) % 6;
        }
    }

    async function handleSideImageInput(image, side) {
         try {
            const formData = new FormData();
            formData.append("image", image);

            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            console.log("Server response:", data);

            if (data.status == 'ok') {
                console.log(data.predictions);

                for(let i = 0; i < 9; i++) {
                    // don't change center color
                    if(i == 4) continue;
                    side[i] = sharedCubeState.color_to_number[data.predictions[i]];
                }
            } else {
                console.error("Processing failed:", data.message);
            }
        } catch (error) {
            window.alert("Upload failed: " + error);
        }

    }

    sketch.mouseClicked = function () {
        checkSideClicked(frontCenterCoordinates(), sharedCubeState.frontSide);
        checkSideClicked(topCenterCoordinates(), sharedCubeState.topSide);
        checkSideClicked(leftCenterCoordinates(), sharedCubeState.leftSide);
        checkSideClicked(rightCenterCoordinates(), sharedCubeState.rightSide);
        checkSideClicked(bottomCenterCoordinates(), sharedCubeState.bottomSide);
        checkSideClicked(backCenterCoordinates(), sharedCubeState.backSide);
    }
};

new p5(s2);