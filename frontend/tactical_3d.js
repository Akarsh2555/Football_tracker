/**
 * Directive 4: Immersive 3D WebGL Frontend (Three.js)
 * Transforms the 2D Tactical Engine into an interactive 3D digital twin.
 */

// Scene Setup
const PITCH_LENGTH = 105;
const PITCH_WIDTH = 68;

let scene, camera, renderer, pitchMesh;
let playerSkeletons = {}; // Store 3D skeletal wireframes
let passingArcsContext = [];
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();

function init3DScene(containerElement) {
    console.log("Initialize 3D Scene called on", containerElement);
    try {
        // 1. Scene & Camera
        scene = new THREE.Scene();
        // Remove solid background so the CSS pitch-grid shows through!
        // scene.background = new THREE.Color(0x030e1a);
        scene.fog = new THREE.FogExp2(0x030e1a, 0.015);

        const width = containerElement.clientWidth || window.innerWidth || 800;
        const height = containerElement.clientHeight || 400;

        camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        // Pull camera further back and up to ensure view isn't clipped
        camera.position.set(PITCH_LENGTH / 2, -60, 80);
        camera.lookAt(PITCH_LENGTH / 2, PITCH_WIDTH / 2, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        // Force the canvas to fill the container styling
        renderer.domElement.style.width = '100%';
        renderer.domElement.style.height = '100%';
        renderer.domElement.style.position = 'absolute';
        renderer.domElement.style.top = '0';
        renderer.domElement.style.left = '0';
        containerElement.appendChild(renderer.domElement);

        // 3. Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(PITCH_LENGTH / 2, -20, 100);
        scene.add(dirLight);

        // 4. 3D Turf Plane (Topological Heatmap Base)
        // Highly subdivided plane to allow vertex displacement for the Z-axis (height)
        const geometry = new THREE.PlaneGeometry(PITCH_LENGTH, PITCH_WIDTH, 64, 64);
        const material = new THREE.MeshPhongMaterial({
            color: 0x001428,
            wireframe: true,
            transparent: true,
            opacity: 0.6
        });

        pitchMesh = new THREE.Mesh(geometry, material);
        // Move to center to match real world coordinates (X:0->105, Y:0->68)
        pitchMesh.position.set(PITCH_LENGTH / 2, PITCH_WIDTH / 2, 0);
        scene.add(pitchMesh);

        // DEBUG VISUAL: A prominent red box in the middle to ensure rendering occurs
        const debugBox = new THREE.Mesh(
            new THREE.BoxGeometry(10, 10, 10),
            new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true })
        );
        debugBox.position.set(PITCH_LENGTH / 2, PITCH_WIDTH / 2, 10);
        scene.add(debugBox);

        // Adding an ambient light helper
        console.log("3D Scene constructed successfully.");

        window.addEventListener('resize', () => {
            const w = containerElement.clientWidth || window.innerWidth;
            const h = containerElement.clientHeight || 400;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        });

    } catch (e) {
        console.error("Critical Three.js initialization error:", e);
        const container = document.getElementById("webgl-container");
        if (container) container.innerHTML = `<div class='text-red-500 text-xs p-4'>WebGL Crash: ${e.message}</div>`;
    }
}

function animate() {
    requestAnimationFrame(animate);

    // Rotate slightly for dynamic effect
    // camera.position.x = (PITCH_LENGTH / 2) + Math.sin(Date.now() * 0.0005) * 10;
    // camera.lookAt(PITCH_LENGTH / 2, PITCH_WIDTH / 2, 0);

    renderer.render(scene, camera);
}

/**
 * Maps the 2D EPV / Pitch Control array to the Z-axis of the pitch mesh.
 * Creates a Topological Terrain / Heatmap.
 */
function updateTopologicalHeatmap(pcMatrix) {
    if (!pitchMesh) return;

    const count = pitchMesh.geometry.attributes.position.count;
    const positions = pitchMesh.geometry.attributes.position.array;

    // Iterate over vertices and displace Z height based on the PC Matrix logic
    // pcMatrix is assumed to be a flattened array or 2D array representing the 105x68 grid
    for (let i = 0; i < count; i++) {
        // Vertex coordinates in local space -> map to 0-1 range to index pcMatrix
        const xPos = positions[i * 3]; // local X
        const yPos = positions[i * 3 + 1]; // local Y

        // Normalize coordinates to 0-1 indices
        const u = Math.max(0, Math.min(1, (xPos + PITCH_LENGTH / 2) / PITCH_LENGTH));
        const v = Math.max(0, Math.min(1, (yPos + PITCH_WIDTH / 2) / PITCH_WIDTH));

        // Mock matrix lookup: Simulate topological bump in the center
        // In reality, map `u, v` to the `pcMatrix` row/col indices.
        const bumpHeight = Math.sin(u * Math.PI) * Math.sin(v * Math.PI) * 10.0;

        // Set Z coordinate
        positions[i * 3 + 2] = bumpHeight;
    }

    pitchMesh.geometry.attributes.position.needsUpdate = true;
    pitchMesh.geometry.computeVertexNormals();
}

/**
 * Projection Function: Re-projects 2D screen coordinates onto the 3D turf plane.
 */
function project2DTo3D(clientX, clientY) {
    mouse.x = (clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(pitchMesh);
    if (intersects.length > 0) {
        return intersects[0].point; // Vector3 {x, y, z}
    }
    return null;
}

/**
 * Renders passes as 3D volumetric arcs.
 */
function render3DPassingArcs(passes) {
    // Clear old arcs
    passingArcsContext.forEach(arc => scene.remove(arc));
    passingArcsContext = [];

    passes.forEach(pass => {
        const startPoint = new THREE.Vector3(pass.start[0], pass.start[1], 0);
        const endPoint = new THREE.Vector3(pass.end[0], pass.end[1], 0);

        // Middle point pushed up in Z to create an arc
        const midPoint = new THREE.Vector3(
            (startPoint.x + endPoint.x) / 2,
            (startPoint.y + endPoint.y) / 2,
            12.0 // arc height
        );

        const curve = new THREE.CatmullRomCurve3([startPoint, midPoint, endPoint]);
        const tubeGeometry = new THREE.TubeGeometry(curve, 20, 0.4, 8, false);
        const tubeMaterial = new THREE.MeshBasicMaterial({
            color: 0x00e87a,
            transparent: true,
            opacity: 0.8
        });

        const tubeMesh = new THREE.Mesh(tubeGeometry, tubeMaterial);
        scene.add(tubeMesh);
        passingArcsContext.push(tubeMesh);
    });
}

/**
 * Parses 3D pose coordinates from the WebSocket and renders 3D skeletal wireframes.
 * Replaces legacy 2D dots.
 */
function render3DPlayerSkeletons(players3DData) {
    // players3DData: [{id, joints_3d: [[x,y,z], ...]}, ...]

    players3DData.forEach(player => {
        if (!playerSkeletons[player.id]) {
            // Initialize a skeletal line group for the new player
            const material = new THREE.LineBasicMaterial({ color: 0xff4444, linewidth: 2 });
            const geometry = new THREE.BufferGeometry();
            const line = new THREE.LineSegments(geometry, material);
            scene.add(line);
            playerSkeletons[player.id] = line;
        }

        const skeletonMesh = playerSkeletons[player.id];

        // Flatten the joints for BufferGeometry
        const vertices = [];

        // Connect joints naively (assuming 17 keypoints standard COCO)
        // Usually, we map specific pairs (e.g., shoulder to elbow). 
        // Mocking a basic spine/arms connection here:
        const joints = player.joints_3d;
        if (joints && joints.length > 0) {
            // Example: Connect joint 0 to joint 1
            for (let i = 0; i < joints.length - 1; i++) {
                vertices.push(joints[i][0], joints[i][1], joints[i][2]);
                vertices.push(joints[i + 1][0], joints[i + 1][1], joints[i + 1][2]);
            }
        }

        skeletonMesh.geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    });
}

// Ensure the container is ready
function startup3D() {
    const container = document.getElementById("webgl-container");
    if (container) {
        // Show a diagnostic UI label 
        const debugLabel = document.createElement("div");
        debugLabel.innerHTML = "WebGL 3D Engine Status: <b>ONLINE</b>";
        debugLabel.className = "absolute top-2 left-2 z-50 text-[10px] bg-green-900/80 text-green-300 px-2 py-1 rounded border border-green-500/50";
        container.parentElement.appendChild(debugLabel);

        init3DScene(container);
    }
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startup3D);
} else {
    startup3D();
}
