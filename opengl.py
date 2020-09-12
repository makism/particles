""" OpenGL visualization of PSO

In this demo we optimize DeJong's function.

"""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
from joblib import Parallel, delayed
from glumpy import app, gl, gloo, glm


def evaluate_particle(particle, pos):
    curr_fitness = pos[0] ** 2 + pos[1] ** 2
    return particle, pos, curr_fitness


vertex = """
#version 120

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float linewidth;
uniform float antialias;

attribute vec4  fg_color;
attribute vec4  bg_color;
attribute float radius;
attribute vec3  position;

varying float v_pointsize;
varying float v_radius;
varying vec4  v_fg_color;
varying vec4  v_bg_color;
void main (void)
{
    v_radius = radius;
    v_fg_color = fg_color;
    v_bg_color = bg_color;

    gl_Position = projection * view * model * vec4(position,1.0);
    gl_PointSize = 1 * (v_radius + linewidth + 1.5*antialias);
}
"""

fragment = """
#version 120

uniform float linewidth;
uniform float antialias;

varying float v_radius;
varying vec4  v_fg_color;
varying vec4  v_bg_color;

float marker(vec2 P, float size)
{
   const float SQRT_2 = 1.4142135623730951;
   float x = SQRT_2/2 * (P.x - P.y);
   float y = SQRT_2/2 * (P.x + P.y);

   float r1 = max(abs(x)- size/2, abs(y)- size/10);
   float r2 = max(abs(y)- size/2, abs(x)- size/10);
   float r3 = max(abs(P.x)- size/2, abs(P.y)- size/10);
   float r4 = max(abs(P.y)- size/2, abs(P.x)- size/10);
   return min( min(r1,r2), min(r3,r4));
}

void main()
{
    float r = (v_radius + linewidth + 1.5*antialias);
    float t = linewidth/2.0 - antialias;
    float signed_distance = length(gl_PointCoord.xy - vec2(0.5,0.5)) * 2 * r - v_radius;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);

    // Inside shape
    if( signed_distance < 0 ) {
        // Fully within linestroke
        if( border_distance < 0 ) {
            gl_FragColor = v_fg_color;
        } else {
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
        }
    // Outside shape
    } else {
        // Fully within linestroke
        if( border_distance < 0 ) {
            gl_FragColor = v_fg_color;
        } else if( abs(signed_distance) < (linewidth/2.0 + antialias) ) {
            gl_FragColor = vec4(v_fg_color.rgb, v_fg_color.a * alpha);
        } else {
            discard;
        }
    }
}
"""

theta, phi = 90.0, 0.0
sx, sy, sz = 1.5, 1.5, 1.5
tx, ty, tz = 0.0, 0.0, 0.0
window = app.Window(width=800, height=800, color=(1, 1, 1, 1))

n = 400000
DATA = np.zeros((400000, 3))
i = 0
for x in np.arange(-1.0, 1.0, 0.0125):
    for y in np.arange(-1.0, 1.0, 0.0125):
        z = x ** 2 + y ** 2
        DATA[i, :] = x, z, y
        i += 1

VIEW_DATA = DATA

DRAW_DATA = np.unique(DATA, axis=0)

rng = np.random.RandomState(0)

n_particles = 100
n_dims = 3
omega = 0.25
phi_p, phi_g = [0.1, 0.01]
initial_indices = rng.choice(DRAW_DATA.shape[0], size=n_particles, replace=False)
positions = DRAW_DATA[initial_indices, :]
best_positions = positions
fitness_scores = np.zeros((n_particles))
best_fitness_scores = np.zeros((n_particles, 1), dtype=np.float32)
velocities = rng.rand(n_particles, n_dims)
results = Parallel(n_jobs=1)(
    delayed(evaluate_particle)(particle, pos) for particle, pos in enumerate(positions)
)
fitness_scores = np.array([f[2] for f in results], dtype=np.float32)
best_fitness_scores = fitness_scores
global_best_position = np.zeros((n_dims))
global_best_fitness = np.inf

ALL_DATA = np.vstack((VIEW_DATA, positions))
ALL_N = n + n_particles

program = gloo.Program(vertex, fragment, count=ALL_N)
view = np.eye(4, dtype=np.float32)
glm.translate(view, 0, 0, -5)
program["position"] = 0.75 * ALL_DATA
program["radius"] = np.hstack((np.ones((n)) * 1.0, np.ones((n_particles)) * 10.0))
program["fg_color"] = 0, 0, 0, 1
colors = np.vstack((np.ones((n, 4)), 0.25 * np.ones((n_particles, 4))))
colors[:, 3] = 1
program["bg_color"] = colors
program["linewidth"] = 0.0
program["antialias"] = 1.0
program["model"] = np.eye(4, dtype=np.float32)
program["projection"] = np.eye(4, dtype=np.float32)
program["view"] = view


@window.event
def on_draw(dt):
    window.clear()

    global ALL_DATA, ALL_N, n, n_particles
    global rng
    global omega, phi_g, phi_p
    global positions, velocities, fitness_scores, best_positions, best_fitness_scores
    global global_best_position, global_best_fitness

    Rp = rng.rand()
    Rg = rng.rand()
    velocities *= omega
    velocities += (best_positions - positions) * (phi_p * Rp)
    velocities += (global_best_position - positions) * (phi_g * Rg)
    positions += velocities
    results = Parallel(n_jobs=1)(
        delayed(evaluate_particle)(particle, pos)
        for particle, pos in enumerate(positions)
    )
    fitness_scores = np.array([f[2] for f in results], dtype=np.float32)
    diff_indices = np.where(fitness_scores < best_fitness_scores)[0]

    if len(diff_indices) > 0:
        best_positions[diff_indices, :] = positions[diff_indices, :]
        best_fitness_scores[diff_indices] = fitness_scores[diff_indices]

    new_best_fitness_indices = np.where(fitness_scores < global_best_fitness)[0]
    if len(new_best_fitness_indices) > 0:
        new_best_fitness_value = np.min(fitness_scores[new_best_fitness_indices])
        new_best_fitness_arg = np.argmin(fitness_scores[new_best_fitness_indices])

        global_best_fitness = new_best_fitness_value
        global_best_position = positions[new_best_fitness_arg]

    #
    #
    #

    global program
    ALL_DATA = np.vstack((VIEW_DATA, positions))
    program["position"] = 0.75 * ALL_DATA
    #
    #
    #

    global theta, phi, translate
    global sx, sy, sz
    window.clear()
    program.draw(gl.GL_POINTS)
    model = np.eye(4, dtype=np.float32)

    glm.scale(model, sx, sy, sz)
    glm.rotate(model, theta, 1, 0, 0)
    glm.rotate(model, phi, 0, 1, 0)
    glm.translate(model, tx, ty, tz)
    program["model"] = model


@window.event
def on_mouse_drag(x, y, dx, dy, button):
    if button == 4:
        global tx, ty, tz

        sign_x = np.sign(dx)
        sign_y = np.sign(dy)

        if dx >= 0:
            tx += 0.01
        else:
            tx -= 0.01

        if dy >= 0:
            ty += 0.01
        else:
            ty -= 0.01

    if button == 2:
        global phi, theta

        if dx >= 0:
            phi += 0.5
        else:
            phi -= 0.5

        if dy >= 0:
            theta += 0.5
        else:
            theta -= 0.5


@window.event
def on_mouse_scroll(x, y, dx, dy):
    global sx, sy, sz

    if dy == 1:
        sx += 0.5
        sy += 0.5
        sz += 0.5
    else:
        if sx == 1.0:
            return

        sx -= 0.5
        sy -= 0.5
        sz -= 0.5


@window.event
def on_resize(width, height):
    program["projection"] = glm.perspective(45.0, width / float(height), 1.0, 1000.0)


gl.glEnable(gl.GL_DEPTH_TEST)
app.run(framerate=30)
