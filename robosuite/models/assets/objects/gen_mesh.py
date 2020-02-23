num_mesh = 57
template='<mesh file="meshes/bowl/bowl_hull_%d.stl" name="m%d"/>'
template2='<geom pos="0 0 0" mesh="m%d" type="mesh" solimp="0.998 0.998 0.001" solref="0.001 1" density="100" friction="0.95 0.3 0.1" material="coke" group="1" condim="4"/>'
for i in range(num_mesh):
    #print(template % (i+1, i+1))
    print(template2 % (i + 1, ))

