(() => {
    const container = document.getElementById("earth-background");

    if (!container || typeof THREE === "undefined") {
        console.error("Earth container or Three.js not found.");
        return;
    }

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
        40,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );

    camera.position.z = 8;

    const renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
    });

    renderer.setPixelRatio(
        Math.min(window.devicePixelRatio, 2)
    );

    renderer.setSize(
        window.innerWidth,
        window.innerHeight
    );

    container.appendChild(renderer.domElement);

    // EARTH GROUP
    const globe = new THREE.Group();
    scene.add(globe);

    const earthGeometry = new THREE.SphereGeometry(
        2.15, 64, 64
    );

    const earthTexture = new THREE.TextureLoader().load(
        "https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg"
    );

    const earthMaterial = new THREE.MeshPhongMaterial({
        map: earthTexture,
        color: 0xffffff,
        shininess: 18
    });

    const earth = new THREE.Mesh(
        earthGeometry,
        earthMaterial
    );

    globe.add(earth);

    // ATMOSPHERIC GLOW
    const atmosphere = new THREE.Mesh(
        new THREE.SphereGeometry(2.22, 64, 64),
        new THREE.MeshBasicMaterial({
            color: 0x50cfff,
            transparent: true,
            opacity: 0.12,
            side: THREE.BackSide
        })
    );

    globe.add(atmosphere);

    // LIGHTING
    const ambient = new THREE.AmbientLight(
        0xc6e7ff,
        1.9
    );

    scene.add(ambient);

    const sun = new THREE.DirectionalLight(
        0xffffff,
        3.0
    );

    sun.position.set(5, 3, 5);
    scene.add(sun);

    const rim = new THREE.DirectionalLight(
        0x54dfff,
        1.5
    );

    rim.position.set(-5, 0, -2);
    scene.add(rim);

    // ORBITAL RINGS
    const ringMaterial = new THREE.MeshBasicMaterial({
        color: 0x38c9e0,
        transparent: true,
        opacity: 0.22,
        side: THREE.DoubleSide
    });

    for (const tilt of [0.45, -0.55]) {
        const ring = new THREE.Mesh(
            new THREE.TorusGeometry(
                2.85,
                0.006,
                8,
                150
            ),
            ringMaterial
        );

        ring.rotation.x = tilt;
        globe.add(ring);
    }

    // BACKGROUND STARS
    const starPositions = [];

    for (let i = 0; i < 500; i++) {
        starPositions.push(
            (Math.random() - 0.5) * 35,
            (Math.random() - 0.5) * 24,
            -4 - Math.random() * 15
        );
    }

    const starGeometry = new THREE.BufferGeometry();

    starGeometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(
            starPositions,
            3
        )
    );

    const stars = new THREE.Points(
        starGeometry,
        new THREE.PointsMaterial({
            color: 0x9dcde2,
            size: 0.035,
            transparent: true,
            opacity: 0.75
        })
    );

    scene.add(stars);

    // RESPONSIVE POSITIONING
    function positionGlobe() {
        const mobile = window.innerWidth < 750;

        globe.position.x = mobile ? 1.2 : 3.1;
        globe.position.y = mobile ? 0.5 : 0.2;

        globe.scale.setScalar(
            mobile ? 0.7 : 1
        );
    }

    function resize() {
        camera.aspect =
            window.innerWidth / window.innerHeight;

        camera.updateProjectionMatrix();

        renderer.setSize(
            window.innerWidth,
            window.innerHeight
        );

        positionGlobe();
    }

    window.addEventListener("resize", resize);
    resize();

    const reducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    ).matches;

    // CONTINUOUS ROTATION
    function animate() {
        requestAnimationFrame(animate);

        if (!reducedMotion) {
            earth.rotation.y += 0.0015;
        }

        renderer.render(scene, camera);
    }

    animate();
})();