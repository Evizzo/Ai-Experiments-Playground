class LeafyChickenShooter {
    constructor() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.leaves = [];
        this.lasers = [];
        this.rocks = [];
        this.explosions = [];
        this.score = 0;
        this.chickens = [];
        this.gun = null;
        this.sword = null;
        this.currentWeapon = 'gun';
        this.isSwinging = false;
        this.scoreElement = null;
        this.moveSpeed = 0.15;
        this.jumpForce = 0.2;
        this.gravity = 0.01;
        this.velocity = new THREE.Vector3();
        this.isJumping = false;
        this.isFlying = false;
        this.jumpCount = 0;
        this.maxJumps = 2;
        this.flySpeed = 0.1;
        this.keys = {};
        this.playerPosition = new THREE.Vector3(0, 2, 5);
        this.mouseSensitivity = 0.002;
        this.euler = new THREE.Euler(0, 0, 0, 'YXZ');
        this.isGameMode = false;
        this.sounds = {
            pop: new Audio('https://assets.mixkit.co/active_storage/sfx/212/212-preview.mp3'),
            laser: new Audio('https://assets.mixkit.co/active_storage/sfx/1435/1435-preview.mp3'),
            rock: new Audio('https://assets.mixkit.co/active_storage/sfx/1435/1435-preview.mp3'),
            explosion: new Audio('https://assets.mixkit.co/sfx/preview/mixkit-cinematic-boom-hit-1311.mp3'),
            swing: new Audio('https://assets.mixkit.co/sfx/preview/mixkit-fast-sword-whoosh-2792.mp3'),
            bounce: new Audio('https://assets.mixkit.co/sfx/preview/mixkit-basketball-ball-hard-hit-2093.mp3')
        };
        this.mountains = [];
        this.clouds = [];
        this.lakes = [];
        this.chickenSpeed = 0.05;
        this.chickenHealth = 100;
        this.chickenDamage = 34;
        this.balls = [];
        this.physicsObjects = [];
        this.ballBounceCount = 0;
        this.maxBallBounces = 3;
        this.ballBounceEnergy = 0.6;
        this.ballFriction = 0.95;
        this.ballGravity = 0.05;
        this.ballInitialVelocity = 2.0;
        this.ballRotationSpeed = 0.1;
        this.explosionRadius = 5;
        this.explosionForce = 0.5;
        this.smokeDuration = 3000;
        this.smokeParticles = [];
        this.particleSystems = [];
        this.init();
    }

    init() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.getElementById('game-container').innerHTML = ''; // Clear container
        document.getElementById('game-container').appendChild(this.renderer.domElement);

        this.camera.position.copy(this.playerPosition);
        this.camera.lookAt(0, 2, -10);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);

        this.createScoreDisplay();
        this.createMountains();
        this.createClouds();
        this.createLakes();
        this.createEnhancedGround();
        this.spawnChickens(3);
        this.createWoodenGun();
        this.createSword();
        this.sword.visible = false;
        
        this.scene.add(this.camera);

        window.addEventListener('resize', () => this.onWindowResize());
        
        this.renderer.domElement.addEventListener('mousedown', (e) => {
            if (!this.isGameMode) {
                this.renderer.domElement.requestPointerLock();
                return;
            }
            if (e.button === 0) { // Left click
                this.shoot();
            } else if (e.button === 2) { // Right click
                this.throwBall();
            }
        });

        this.renderer.domElement.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });

        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        document.addEventListener('keyup', (e) => this.onKeyUp(e));
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
        document.addEventListener('pointerlockchange', () => this.onPointerLockChange());

        this.animate();
    }

    createScoreDisplay() {
        const existingScoreElement = document.getElementById('score-display');
        if (existingScoreElement) {
            existingScoreElement.remove();
        }

        this.scoreElement = document.createElement('div');
        this.scoreElement.id = 'score-display';
        Object.assign(this.scoreElement.style, {
            position: 'absolute',
            top: '20px',
            right: '20px',
            padding: '10px 20px',
            backgroundColor: 'rgba(0,0,0,0.5)',
            color: 'white',
            fontFamily: 'Arial, sans-serif',
            fontSize: '24px',
            border: '2px solid white',
            borderRadius: '5px',
            zIndex: '100'
        });
        this.updateScoreDisplay();
        document.body.appendChild(this.scoreElement);
    }

    updateScoreDisplay() {
        if (this.scoreElement) {
            this.scoreElement.textContent = `Score: ${this.score}`;
        }
    }

    createMountains() {
        const mountainCount = 10;
        const mountainGeometry = new THREE.ConeGeometry(5, 15, 4);
        const mountainMaterial = new THREE.MeshStandardMaterial({ color: 0x808080, roughness: 0.9, metalness: 0.1 });

        for (let i = 0; i < mountainCount; i++) {
            const mountain = new THREE.Mesh(mountainGeometry, mountainMaterial);
            const angle = (i / mountainCount) * Math.PI * 2;
            const radius = 40 + Math.random() * 10;
            mountain.position.set( Math.cos(angle) * radius, 0, Math.sin(angle) * radius );
            mountain.rotation.y = Math.random() * Math.PI;
            mountain.scale.set( 1 + Math.random() * 0.5, 1 + Math.random() * 1, 1 + Math.random() * 0.5 );
            mountain.castShadow = true;
            mountain.receiveShadow = true;
            this.mountains.push(mountain);
            this.scene.add(mountain);
        }
    }

    createClouds() {
        const cloudCount = 20;
        const cloudGeometry = new THREE.SphereGeometry(2, 8, 8);
        const cloudMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });

        for (let i = 0; i < cloudCount; i++) {
            const cloud = new THREE.Group();
            const cloudPieces = 3 + Math.floor(Math.random() * 3);
            
            for (let j = 0; j < cloudPieces; j++) {
                const piece = new THREE.Mesh(cloudGeometry, cloudMaterial);
                piece.position.set( Math.random() * 4 - 2, Math.random() * 2, Math.random() * 4 - 2 );
                piece.scale.set( 1 + Math.random() * 0.5, 1 + Math.random() * 0.5, 1 + Math.random() * 0.5 );
                cloud.add(piece);
            }

            const angle = Math.random() * Math.PI * 2;
            const radius = 30 + Math.random() * 20;
            cloud.position.set( Math.cos(angle) * radius, 20 + Math.random() * 10, Math.sin(angle) * radius );
            cloud.userData.speed = 0.01 + Math.random() * 0.02;
            this.clouds.push(cloud);
            this.scene.add(cloud);
        }
    }

    createLakes() {
        const lakeCount = 3;
        const lakeGeometry = new THREE.CircleGeometry(5, 32);
        const lakeMaterial = new THREE.MeshStandardMaterial({ color: 0x0077be, transparent: true, opacity: 0.8, metalness: 0.9, roughness: 0.1 });

        for (let i = 0; i < lakeCount; i++) {
            const lake = new THREE.Mesh(lakeGeometry, lakeMaterial);
            const angle = (i / lakeCount) * Math.PI * 2;
            const radius = 25 + Math.random() * 10;
            lake.position.set( Math.cos(angle) * radius, 0.1, Math.sin(angle) * radius );
            lake.rotation.x = -Math.PI / 2;
            lake.scale.set( 1 + Math.random() * 0.5, 1 + Math.random() * 0.5, 1 );
            this.lakes.push(lake);
            this.scene.add(lake);
        }
    }

    createEnhancedGround() {
        const groundSize = 100;
        const groundGeometry = new THREE.PlaneGeometry(groundSize, groundSize, 32, 32);
        const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x1a472a, roughness: 0.8, metalness: 0.2 });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -0.5;
        ground.receiveShadow = true;
        this.scene.add(ground);
        this.addDecorativeElements();
    }

    addDecorativeElements() {
        for (let i = 0; i < 20; i++) {
            const tree = this.createTree();
            tree.position.set( (Math.random() - 0.5) * 80, 0, (Math.random() - 0.5) * 80 );
            this.scene.add(tree);
        }
        for (let i = 0; i < 30; i++) {
            const rock = this.createRock();
            rock.position.set( (Math.random() - 0.5) * 90, 0, (Math.random() - 0.5) * 90 );
            this.scene.add(rock);
        }
    }

    createTree() {
        const tree = new THREE.Group();
        
        const trunkGeometry = new THREE.CylinderGeometry(0.5, 0.7, 5, 8);
        const trunkMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x4d2926, 
            roughness: 0.9,
            bumpScale: 0.1,
            metalness: 0.1
        });
        const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
        trunk.position.y = 2.5;
        trunk.castShadow = true;
        trunk.receiveShadow = true;
        tree.add(trunk);

        const leavesLayers = 3;
        for (let i = 0; i < leavesLayers; i++) {
            const leavesGeometry = new THREE.ConeGeometry(2 - i * 0.3, 3, 8);
            const leavesMaterial = new THREE.MeshStandardMaterial({ 
                color: 0x2d5a27, 
                roughness: 0.8,
                metalness: 0.1
            });
            const leaves = new THREE.Mesh(leavesGeometry, leavesMaterial);
            leaves.position.y = 5 + i * 1.5;
            leaves.castShadow = true;
            leaves.receiveShadow = true;
            tree.add(leaves);
        }

        const collisionBox = new THREE.Box3().setFromObject(tree);
        tree.userData.collisionBox = collisionBox;
        this.physicsObjects.push(tree);

        return tree;
    }

    createRock() {
        const rockGeometry = new THREE.DodecahedronGeometry(Math.random() * 0.5 + 0.5, 1);
        const rockMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x808080, 
            roughness: 0.9, 
            metalness: 0.1,
            bumpScale: 0.2
        });
        const rock = new THREE.Mesh(rockGeometry, rockMaterial);
        rock.position.y = 0.5;
        rock.rotation.set(
            Math.random() * Math.PI,
            Math.random() * Math.PI,
            Math.random() * Math.PI
        );
        rock.castShadow = true;
        rock.receiveShadow = true;

        const collisionBox = new THREE.Box3().setFromObject(rock);
        rock.userData.collisionBox = collisionBox;
        this.physicsObjects.push(rock);

        return rock;
    }

    createWoodenGun() {
        const gun = new THREE.Group();
        const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x8B4513, roughness: 0.9 });
        const bodyGeometry = new THREE.CylinderGeometry(0.05, 0.05, 1, 8);
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.rotation.x = Math.PI / 2;
        body.position.z = -0.5;
        gun.add(body);
        const handleGeometry = new THREE.CylinderGeometry(0.03, 0.03, 0.3, 8);
        const handle = new THREE.Mesh(handleGeometry, bodyMaterial);
        handle.rotation.x = Math.PI / 2;
        handle.rotation.z = Math.PI / 2;
        handle.position.set(0.1, -0.1, -0.3);
        gun.add(handle);
        const barrelGeometry = new THREE.CylinderGeometry(0.07, 0.05, 0.2, 8);
        const barrel = new THREE.Mesh(barrelGeometry, bodyMaterial);
        barrel.rotation.x = Math.PI / 2;
        barrel.position.z = -0.9;
        gun.add(barrel);
        gun.position.set(0.3, -0.2, -0.5);
        this.gun = gun;
        this.camera.add(gun);
    }
    
    createSword() {
        const sword = new THREE.Group();
        const silverMaterial = new THREE.MeshStandardMaterial({ color: 0xC0C0C0, metalness: 0.8, roughness: 0.4 });
        const goldMaterial = new THREE.MeshStandardMaterial({ color: 0xFFD700, metalness: 0.7, roughness: 0.5 });
        const bladeGeometry = new THREE.BoxGeometry(0.05, 1.2, 0.2);
        const blade = new THREE.Mesh(bladeGeometry, silverMaterial);
        blade.position.y = 0.6;
        sword.add(blade);
        const guardGeometry = new THREE.BoxGeometry(0.08, 0.08, 0.5);
        const guard = new THREE.Mesh(guardGeometry, goldMaterial);
        sword.add(guard);
        const hiltGeometry = new THREE.CylinderGeometry(0.04, 0.04, 0.3, 8);
        const hilt = new THREE.Mesh(hiltGeometry, goldMaterial);
        hilt.position.y = -0.19;
        sword.add(hilt);
        const pommelGeometry = new THREE.SphereGeometry(0.06, 8, 8);
        const pommel = new THREE.Mesh(pommelGeometry, goldMaterial);
        pommel.position.y = -0.38;
        sword.add(pommel);
        sword.position.set(0.4, -0.4, -0.8);
        sword.rotation.x = Math.PI / 8;
        sword.rotation.z = -Math.PI / 16;
        this.sword = sword;
        this.camera.add(this.sword);
    }

    onKeyDown(event) {
        this.keys[event.code] = true;
        if (event.code === 'Digit1') {
            this.currentWeapon = 'gun';
            this.gun.visible = true;
            this.sword.visible = false;
        } else if (event.code === 'Digit2') {
            this.currentWeapon = 'sword';
            this.gun.visible = false;
            this.sword.visible = true;
        }
        if (event.code === 'Space') {
            if (this.jumpCount < this.maxJumps) {
                this.velocity.y = this.jumpForce;
                this.jumpCount++;
                if (this.jumpCount === this.maxJumps) {
                    this.isFlying = true;
                }
            }
        }
    }

    onKeyUp(event) {
        this.keys[event.code] = false;
    }

    onMouseMove(event) {
        if (!this.isGameMode) return;
        const movementX = event.movementX || 0;
        const movementY = event.movementY || 0;
        this.euler.y -= movementX * this.mouseSensitivity;
        this.euler.x -= movementY * this.mouseSensitivity;
        this.euler.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.euler.x));
        this.camera.quaternion.setFromEuler(this.euler);
    }

    onPointerLockChange() {
        this.isGameMode = document.pointerLockElement === this.renderer.domElement;
    }

    spawnChickens(count) {
        for (let i = 0; i < count; i++) {
            this.spawnChicken();
        }
    }

    spawnChicken() {
        const chicken = this.createChicken();
        const angle = Math.random() * Math.PI * 2;
        const radius = 15 + Math.random() * 10;
        chicken.position.set( Math.cos(angle) * radius, 0, Math.sin(angle) * radius );
        this.chickens.push(chicken);
        this.scene.add(chicken);
    }

    createChicken() {
        const chicken = new THREE.Group();
        const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8 });
        const bodyGeometry = new THREE.SphereGeometry(1, 32, 32);
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.castShadow = true;
        chicken.add(body);
        const headGeometry = new THREE.SphereGeometry(0.5, 32, 32);
        const head = new THREE.Mesh(headGeometry, bodyMaterial);
        head.position.set(0, 1, 0.5);
        head.castShadow = true;
        chicken.add(head);
        const beakGeometry = new THREE.ConeGeometry(0.2, 0.5, 32);
        const beakMaterial = new THREE.MeshStandardMaterial({ color: 0xffd700, roughness: 0.5 });
        const beak = new THREE.Mesh(beakGeometry, beakMaterial);
        beak.position.set(0, 1, 1);
        beak.rotation.x = -Math.PI / 2;
        beak.castShadow = true;
        chicken.add(beak);
        const legMaterial = new THREE.MeshStandardMaterial({ color: 0xffd700, roughness: 0.5 });
        const legGeometry = new THREE.CylinderGeometry(0.1, 0.1, 1, 8);
        const leftLeg = new THREE.Mesh(legGeometry, legMaterial);
        leftLeg.position.set(-0.3, -1, 0);
        leftLeg.castShadow = true;
        chicken.add(leftLeg);
        const rightLeg = new THREE.Mesh(legGeometry, legMaterial);
        rightLeg.position.set(0.3, -1, 0);
        rightLeg.castShadow = true;
        chicken.add(rightLeg);
        const healthBarGeometry = new THREE.PlaneGeometry(1, 0.1);
        const healthBarMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.8, side: THREE.DoubleSide });
        const healthBar = new THREE.Mesh(healthBarGeometry, healthBarMaterial);
        healthBar.position.set(0, 2, 0);
        chicken.add(healthBar);
        chicken.children.forEach(child => { child.userData.chicken = chicken; });
        chicken.userData.healthBar = healthBar;
        chicken.userData.health = this.chickenHealth;
        chicken.userData.speed = 0.02 + Math.random() * 0.02;
        chicken.userData.direction = new THREE.Vector3(Math.random() - 0.5, 0, Math.random() - 0.5).normalize();
        return chicken;
    }

    updateChickens() {
        for (let i = this.chickens.length - 1; i >= 0; i--) {
            const chicken = this.chickens[i];
            const distanceToPlayer = chicken.position.distanceTo(this.playerPosition);
            if (distanceToPlayer < 10) {
                const directionToPlayer = new THREE.Vector3().subVectors(this.playerPosition, chicken.position).normalize();
                chicken.userData.direction.copy(directionToPlayer).multiplyScalar(-1);
            } else if (Math.random() < 0.02) {
                chicken.userData.direction.set(Math.random() - 0.5, 0, Math.random() - 0.5).normalize();
            }
            const moveStep = chicken.userData.direction.clone().multiplyScalar(chicken.userData.speed);
            chicken.position.add(moveStep);
            const maxDistance = 40;
            if (chicken.position.length() > maxDistance) {
                chicken.userData.direction.multiplyScalar(-1);
            }
            chicken.lookAt(chicken.position.clone().add(chicken.userData.direction));
            chicken.position.y = Math.sin(Date.now() * 0.003 + i) * 0.1;
            if (chicken.userData.healthBar) {
                chicken.userData.healthBar.position.copy(chicken.position);
                chicken.userData.healthBar.position.y += 2.5;
                chicken.userData.healthBar.lookAt(this.camera.position);
            }
        }
    }

    shoot() {
        if (!this.isGameMode) return;
        switch (this.currentWeapon) {
            case 'gun':
                if (this.isFlying) { this.shootLaser(); } else { this.shootLeaf(); }
                break;
            case 'sword':
                this.meleeAttack();
                break;
        }
    }

    meleeAttack() {
        if (this.isSwinging) return;
        this.isSwinging = true;
        this.sounds.swing.currentTime = 0;
        this.sounds.swing.play();
        const originalRotation = this.sword.rotation.clone();
        setTimeout(() => { this.sword.rotation.z -= Math.PI / 4; }, 0);
        setTimeout(() => { this.sword.rotation.z += Math.PI / 2; }, 80);
        const meleeRange = 3.5;
        const meleeAngle = 0.7;
        const meleeDamage = 50;
        const cameraDirection = new THREE.Vector3();
        this.camera.getWorldDirection(cameraDirection);
        for (let i = this.chickens.length - 1; i >= 0; i--) {
            const chicken = this.chickens[i];
            const directionToChicken = chicken.position.clone().sub(this.playerPosition).normalize();
            const distance = this.playerPosition.distanceTo(chicken.position);
            const dot = cameraDirection.dot(directionToChicken);
            if (distance < meleeRange && dot > meleeAngle) {
                chicken.userData.health -= meleeDamage;
                this.checkChickenHealth(chicken);
            }
        }
        setTimeout(() => {
            this.sword.rotation.copy(originalRotation);
            this.isSwinging = false;
        }, 250);
    }
    
    checkChickenHealth(chicken) {
        const healthPercent = Math.max(0, chicken.userData.health / this.chickenHealth);
        chicken.userData.healthBar.scale.x = healthPercent;
        if (healthPercent < 0.3) {
            chicken.userData.healthBar.material.color.setHex(0xff0000);
        } else if (healthPercent < 0.6) {
            chicken.userData.healthBar.material.color.setHex(0xffff00);
        }
        if (chicken.userData.health <= 0) {
            this.sounds.pop.currentTime = 0;
            this.sounds.pop.play();
            this.scene.remove(chicken);
            const index = this.chickens.indexOf(chicken);
            if (index > -1) { this.chickens.splice(index, 1); }
            this.score += 10;
            this.updateScoreDisplay();
            setTimeout(() => this.spawnChicken(), 2000);
        }
    }

    shootLeaf() {
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);
        const intersects = raycaster.intersectObjects(this.scene.children, true);
        if (intersects.length > 0 && intersects[0].object.userData.chicken) {
            const hitChicken = intersects[0].object.userData.chicken;
            hitChicken.userData.health -= this.chickenDamage;
            this.checkChickenHealth(hitChicken);
        }
        const gunWorldPosition = new THREE.Vector3();
        this.gun.getWorldPosition(gunWorldPosition);
        const leaf = this.createLeaf();
        leaf.position.copy(gunWorldPosition);
        const direction = raycaster.ray.direction;
        leaf.userData.velocity = direction.clone().multiplyScalar(0.8);
        this.scene.add(leaf);
        this.leaves.push(leaf);
    }

    createLeaf() {
        const leafGeometry = new THREE.PlaneGeometry(0.2, 0.4);
        const leafMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff00, side: THREE.DoubleSide, roughness: 0.8 });
        const leaf = new THREE.Mesh(leafGeometry, leafMaterial);
        return leaf;
    }

    updateLeaves() {
        for (let i = this.leaves.length - 1; i >= 0; i--) {
            const leaf = this.leaves[i];
            leaf.position.add(leaf.userData.velocity);
            leaf.rotation.z += 0.1;
            if (leaf.position.distanceTo(this.camera.position) > 100) {
                this.scene.remove(leaf);
                this.leaves.splice(i, 1);
            }
        }
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    updateMovement() {
        if (!this.isGameMode) return;
        const horizontalVelocity = new THREE.Vector3();
        const speed = this.moveSpeed;
        const forward = new THREE.Vector3();
        this.camera.getWorldDirection(forward);
        forward.y = 0;
        forward.normalize();
        const right = new THREE.Vector3();
        right.crossVectors(this.camera.up, forward).negate();
        if (this.keys['KeyW'] || this.keys['ArrowUp']) { horizontalVelocity.add(forward); }
        if (this.keys['KeyS'] || this.keys['ArrowDown']) { horizontalVelocity.sub(forward); }
        if (this.keys['KeyA'] || this.keys['ArrowLeft']) { horizontalVelocity.sub(right); }
        if (this.keys['KeyD'] || this.keys['ArrowRight']) { horizontalVelocity.add(right); }
        if (horizontalVelocity.lengthSq() > 0) {
            horizontalVelocity.normalize();
            this.playerPosition.add(horizontalVelocity.multiplyScalar(speed));
        }
        if (this.isFlying) {
            if (this.keys['Space']) { this.velocity.y = this.flySpeed; } 
            else if (this.keys['ShiftLeft']) { this.velocity.y = -this.flySpeed; } 
            else { this.velocity.y = 0; }
        } else {
            this.velocity.y -= this.gravity;
        }
        this.playerPosition.y += this.velocity.y;
        if (this.playerPosition.y < 2) {
            this.playerPosition.y = 2;
            this.velocity.y = 0;
            this.isJumping = false;
            this.jumpCount = 0;
            this.isFlying = false;
        }
        this.camera.position.copy(this.playerPosition);
    }

    shootLaser() {
        if (!this.isGameMode) return;
        this.sounds.laser.currentTime = 0;
        this.sounds.laser.play();
        const laserMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending });
        const laserGeometry = new THREE.CylinderGeometry(0.05, 0.05, 50, 8);
        const laser = new THREE.Mesh(laserGeometry, laserMaterial);
        const glowMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.3, blending: THREE.AdditiveBlending });
        const glowGeometry = new THREE.CylinderGeometry(0.1, 0.1, 50, 8);
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        laser.add(glow);
        const gunWorldPosition = new THREE.Vector3();
        this.gun.getWorldPosition(gunWorldPosition);
        laser.position.copy(gunWorldPosition);
        const direction = new THREE.Vector3();
        this.camera.getWorldDirection(direction);
        laser.lookAt(direction.multiplyScalar(100).add(laser.position));
        laser.rotateX(Math.PI / 2);
        this.scene.add(laser);
        this.lasers.push(laser);
        const particles = new THREE.Group();
        for (let i = 0; i < 10; i++) {
            const particleMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending });
            const particleGeometry = new THREE.SphereGeometry(0.1, 8, 8);
            const particle = new THREE.Mesh(particleGeometry, particleMaterial);
            particle.position.set( Math.random() * 0.5 - 0.25, Math.random() * 0.5 - 0.25, Math.random() * 0.5 - 0.25 );
            particles.add(particle);
        }
        particles.position.copy(laser.position).add(direction.multiplyScalar(25));
        this.scene.add(particles);
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);
        const intersects = raycaster.intersectObjects(this.scene.children, true);
        if (intersects.length > 0 && intersects[0].object.userData.chicken) {
            const hitChicken = intersects[0].object.userData.chicken;
            hitChicken.userData.health -= 100;
            this.checkChickenHealth(hitChicken);
        }
        setTimeout(() => {
            this.scene.remove(laser);
            this.scene.remove(particles);
            this.lasers.splice(this.lasers.indexOf(laser), 1);
        }, 100);
    }

    throwRock() {
        if (!this.isGameMode) return;
        this.sounds.rock.currentTime = 0;
        this.sounds.rock.play();
        const rockGeometry = new THREE.SphereGeometry(0.3, 16, 16);
        const rockMaterial = new THREE.MeshStandardMaterial({ color: 0x808080, roughness: 0.9 });
        const rock = new THREE.Mesh(rockGeometry, rockMaterial);
        const gunWorldPosition = new THREE.Vector3();
        this.gun.getWorldPosition(gunWorldPosition);
        rock.position.copy(gunWorldPosition);
        const direction = new THREE.Vector3();
        this.camera.getWorldDirection(direction);
        rock.userData.velocity = direction.clone().multiplyScalar(0.5);
        rock.userData.velocity.y += 0.2;
        rock.userData.bounces = 0;
        rock.userData.maxBounces = 2;
        this.scene.add(rock);
        this.rocks.push(rock);
    }

    updateRocks() {
        for (let i = this.rocks.length - 1; i >= 0; i--) {
            const rock = this.rocks[i];
            let exploded = false;
            rock.userData.velocity.y -= this.gravity;
            rock.position.add(rock.userData.velocity);
            rock.rotation.x += 0.1;
            rock.rotation.z += 0.1;
            for (let j = this.chickens.length - 1; j >= 0; j--) {
                const chicken = this.chickens[j];
                if (rock.position.distanceTo(chicken.position) < 1.5) {
                    this.createExplosion(rock.position);
                    exploded = true;
                    break; 
                }
            }
            if (exploded) {
                this.scene.remove(rock);
                this.rocks.splice(i, 1);
                continue;
            }
            if (rock.position.y < -0.2) {
                rock.userData.bounces++;
                if (rock.userData.bounces >= rock.userData.maxBounces) {
                    this.createExplosion(rock.position);
                    exploded = true;
                } else {
                    rock.position.y = -0.2;
                    rock.userData.velocity.y *= -0.6;
                }
            }
            if (exploded) {
                this.scene.remove(rock);
                this.rocks.splice(i, 1);
            }
        }
    }

    createExplosion(position) {
        this.sounds.explosion.currentTime = 0;
        this.sounds.explosion.play();
        const explosion = new THREE.Group();
        explosion.position.copy(position);
        const particleCount = 20;
        const particleGeometry = new THREE.SphereGeometry(0.1, 8, 8);
        for(let i = 0; i < particleCount; i++) {
            const particleMaterial = new THREE.MeshBasicMaterial({ 
                color: Math.random() > 0.5 ? 0xff4500 : 0xffa500, 
                transparent: true,
                opacity: 0.8 
            });
            const particle = new THREE.Mesh(particleGeometry, particleMaterial);
            particle.userData.velocity = new THREE.Vector3(
                (Math.random() - 0.5) * 0.2,
                (Math.random() - 0.5) * 0.2,
                (Math.random() - 0.5) * 0.2
            );
            explosion.add(particle);
        }
        this.scene.add(explosion);
        this.explosions.push({ group: explosion, startTime: Date.now(), duration: 500 });

        const blastRadius = 3;
        const explosionDamage = 50;
        for (let i = this.chickens.length - 1; i >= 0; i--) {
            const chicken = this.chickens[i];
            const distance = chicken.position.distanceTo(position);
            if (distance < blastRadius) {
                chicken.userData.health -= explosionDamage;
                this.checkChickenHealth(chicken);
            }
        }
    }
    
    updateExplosions() {
        for (let i = this.explosions.length - 1; i >= 0; i--) {
            const explosionData = this.explosions[i];
            const elapsed = Date.now() - explosionData.startTime;
            const progress = elapsed / explosionData.duration;
            if (progress >= 1) {
                this.scene.remove(explosionData.group);
                this.explosions.splice(i, 1);
            } else {
                explosionData.group.children.forEach(particle => {
                    particle.position.add(particle.userData.velocity);
                    particle.material.opacity = 1.0 - progress;
                });
            }
        }
    }

    updateClouds() {
        this.clouds.forEach(cloud => {
            cloud.position.x += cloud.userData.speed;
            if (cloud.position.x > 50) {
                cloud.position.x = -50;
            }
        });
    }

    throwBall() {
        if (!this.isGameMode) return;

        const ballGeometry = new THREE.SphereGeometry(0.3, 32, 32);
        const ballMaterial = new THREE.MeshStandardMaterial({
            color: 0xff4500,
            metalness: 0.8,
            roughness: 0.2,
            emissive: 0xff4500,
            emissiveIntensity: 0.2
        });

        const ball = new THREE.Mesh(ballGeometry, ballMaterial);
        
        // Get camera position and direction
        const cameraPosition = this.camera.position.clone();
        const cameraDirection = new THREE.Vector3(0, 0, -1);
        cameraDirection.applyQuaternion(this.camera.quaternion);
        
        // Set initial position in front of camera
        ball.position.copy(cameraPosition).add(cameraDirection.multiplyScalar(2));
        
        // Calculate initial velocity with more force
        const throwDirection = cameraDirection.clone();
        throwDirection.y += 0.3; // Add more upward force
        throwDirection.normalize();
        
        // Set initial velocity with higher value
        const initialVelocity = throwDirection.multiplyScalar(2.0);
        
        ball.userData = {
            velocity: initialVelocity,
            rotation: new THREE.Vector3(
                Math.random() * 0.1,
                Math.random() * 0.1,
                Math.random() * 0.1
            ),
            bounces: 0,
            lastCollision: null,
            trail: []
        };

        this.balls.push(ball);
        this.scene.add(ball);
        this.sounds.bounce.play();
    }

    updateBalls() {
        for (let i = this.balls.length - 1; i >= 0; i--) {
            const ball = this.balls[i];
            
            // Store current position for trail
            ball.userData.trail.push(ball.position.clone());
            if (ball.userData.trail.length > 10) {
                ball.userData.trail.shift();
            }
            
            // Apply stronger gravity
            ball.userData.velocity.y -= 0.05;
            
            // Apply friction
            ball.userData.velocity.multiplyScalar(0.95);
            
            // Update position
            ball.position.add(ball.userData.velocity);
            
            // Rotate ball based on velocity
            ball.rotation.x += ball.userData.velocity.x * 0.1;
            ball.rotation.y += ball.userData.velocity.y * 0.1;
            ball.rotation.z += ball.userData.velocity.z * 0.1;

            // Create a bounding sphere for the ball
            const ballCollider = new THREE.Sphere(ball.position, ball.geometry.parameters.radius);

            // Check collisions with physics objects
            let hasCollided = false;
            for (const obj of this.physicsObjects) {
                if (obj.userData.collisionBox && obj.userData.collisionBox.intersectsSphere(ballCollider)) {
                    hasCollided = true;
                    this.handleBallCollision(ball, obj);
                    break;
                }
            }

            // Check ground collision
            if (ball.position.y <= ball.geometry.parameters.radius) {
                hasCollided = true;
                this.handleBallCollision(ball, null);
            }

            // Check bounce limit and create explosion
            if (ball.userData.bounces >= this.maxBallBounces) {
                this.createExplosion(ball.position.clone());
                this.scene.remove(ball);
                this.balls.splice(i, 1);
                continue;
            }

            // Create trail effect
            this.createBallTrail(ball);
        }
    }

    createBallTrail(ball) {
        if (ball.userData.trail.length < 2) return;

        const trailGeometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];

        for (let i = 0; i < ball.userData.trail.length; i++) {
            const position = ball.userData.trail[i];
            positions.push(position.x, position.y, position.z);
            
            const alpha = i / ball.userData.trail.length;
            colors.push(1, 0.27, 0, alpha);
        }

        trailGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        trailGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 4));

        const trailMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.5
        });

        const trail = new THREE.Line(trailGeometry, trailMaterial);
        this.scene.add(trail);
        
        setTimeout(() => {
            this.scene.remove(trail);
        }, 100);
    }

    handleBallCollision(ball, collider) {
        // Create explosion at ball's position
        this.createExplosion(ball.position.clone());
        
        // Remove the ball from the scene and array
        this.scene.remove(ball);
        const ballIndex = this.balls.indexOf(ball);
        if (ballIndex > -1) {
            this.balls.splice(ballIndex, 1);
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (this.isGameMode) {
            this.updateMovement();
            this.updateChickens();
            this.updateLeaves();
            this.updateRocks();
            this.updateExplosions();
            this.updateClouds();
            this.updateBalls(); // Make sure this is called
        }
        
        this.renderer.render(this.scene, this.camera);
    }
}

window.addEventListener('load', () => {
    new LeafyChickenShooter();
});
