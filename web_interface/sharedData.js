let sharedCubeState = {
    topSide: [0,0,0,0,0,0,0,0,0],
    bottomSide: [0,0,0,0,5,0,0,0,0],
    frontSide: [0,0,0,0,2,0,0,0,0],
    leftSide: [0,0,0,0,4,0,0,0,0],
    rightSide: [0,0,0,0,1,0,0,0,0],
    backSide: [0,0,0,0,3,0,0,0,0],
    
    color_dict: {
        0: [255, 255, 255], //white
        1: [255, 0, 0], // red
        2: [0, 255, 0], // green
        3: [0, 0, 255], // blue
        4: [255, 165, 0], // orange
        5: [255, 255, 0], // yellow
    },

    number_to_color: {
        0: "white",
        1: "red",
        2: "green",
        3: "blue",
        4: "orange",
        5: "yellow" 
    },

    color_to_number: {
        "white": 0,
        "red": 1,
        "green": 2,
        "blue": 3,
        "orange": 4,
        "yellow": 5 
    }
};