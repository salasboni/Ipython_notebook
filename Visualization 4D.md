

    %matplotlib inline
    
    import itertools as it
    import numpy as np, scipy as sp, matplotlib.pyplot as plt



    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import numpy as np
    
    n_angles = 36
    n_radii = 8
    
    # An array of radii
    # Does not include radius r=0, this is to eliminate duplicate points
    radii = np.linspace(0.125, 1.0, n_radii)
    
    # An array of angles
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    
    # Repeat all angles for each radius
    angles = np.repeat(angles[...,np.newaxis], n_radii, axis=1)
    
    # Convert polar (radii, angles) coords to cartesian (x, y) coords
    # (0, 0) is added here. There are no duplicate points in the (x, y) plane
    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())
    
    # Pringle surface
    z = np.sin(-x*y)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    
    plt.show()


![png](Visualization%204D_files/Visualization%204D_1_0.png)



    def randrange(n, vmin, vmax):
        return (vmax-vmin)*np.random.rand(n) + vmin
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zl, zh)
        ax.scatter(xs, ys, zs, c=c, s=100, marker=m)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()



![png](Visualization%204D_files/Visualization%204D_2_0.png)



    x = range(1,10)
    print(plt.cm.hot(x))

    [[ 0.05189484  0.          0.          1.        ]
     [ 0.06218969  0.          0.          1.        ]
     [ 0.07248453  0.          0.          1.        ]
     [ 0.08277938  0.          0.          1.        ]
     [ 0.09307422  0.          0.          1.        ]
     [ 0.10336906  0.          0.          1.        ]
     [ 0.11366391  0.          0.          1.        ]
     [ 0.12395875  0.          0.          1.        ]
     [ 0.1342536   0.          0.          1.        ]]



    plt.scatter(x, x, color = plt.cm.hot(x))




    <matplotlib.collections.PathCollection at 0x105d20610>




![png](Visualization%204D_files/Visualization%204D_4_1.png)



    import matplotlib as mpl
    import matplotlib.cm as cm
    
    norm = mpl.colors.Normalize(vmin=-20, vmax=10)
    cmap = cm.hot
    x = 0.3
    
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    print m.to_rgba(x)

    (1.0, 0.82254864129963445, 0.0, 1.0)



    # x = np.arange(0, 10, .1)
    # y = np.arange(0, 10, .1)
    # X, Y = np.meshgrid(x,y)
    dx, dy = 0.15, 0.05
    
    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(-3, 3 + dy, dy),
                    slice(-3, 3 + dx, dx)]
    z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    plt.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    plt.title('pcolor')
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()
    
    # plt.pcolor(x, t, z, cmap=cm)
    # plt.clim(-4,4)
    # plt.show()




    <matplotlib.colorbar.Colorbar instance at 0x10dfb4a70>




![png](Visualization%204D_files/Visualization%204D_6_1.png)



    # Just plotting the values of data that are nonzero 
    
    data = np.random.rand(10,2)
    
    x_data = data[0] # x coordinates
    y_data = data[1]# y coordinates
    
    # Mapping the values to RGBA colors
    # pts = plt.scatter(x_data, y_data, marker='s', c=[x_data, y_data])
    # data = plt.cm.jet([x_data, y_data])
    
    pts = plt.scatter(x_data, y_data, marker='s', c=data)
    plt.colorbar(pts)


    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-46-dc7a7b5f7ef2> in <module>()
         10 # data = plt.cm.jet([x_data, y_data])
         11 
    ---> 12 pts = plt.scatter(x_data, y_data, marker='s', c=data)
         13 plt.colorbar(pts)


    /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/matplotlib/pyplot.pyc in scatter(x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, hold, **kwargs)
       3085         ret = ax.scatter(x, y, s=s, c=c, marker=marker, cmap=cmap, norm=norm,
       3086                          vmin=vmin, vmax=vmax, alpha=alpha,
    -> 3087                          linewidths=linewidths, verts=verts, **kwargs)
       3088         draw_if_interactive()
       3089     finally:


    /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/matplotlib/axes.pyc in scatter(self, x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, **kwargs)
       6276                 colors = None  # use cmap, norm after collection is created
       6277             else:
    -> 6278                 colors = mcolors.colorConverter.to_rgba_array(c, alpha)
       6279 
       6280         faceted = kwargs.pop('faceted', None)


    /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/matplotlib/colors.pyc in to_rgba_array(self, c, alpha)
        409             result = np.zeros((nc, 4), dtype=np.float)
        410             for i, cc in enumerate(c):
    --> 411                 result[i] = self.to_rgba(cc, alpha)
        412             return result
        413 


    /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/matplotlib/colors.pyc in to_rgba(self, arg, alpha)
        363         except (TypeError, ValueError) as exc:
        364             raise ValueError(
    --> 365                 'to_rgba: Invalid rgba arg "%s"\n%s' % (str(arg), exc))
        366 
        367     def to_rgba_array(self, c, alpha=None):


    ValueError: to_rgba: Invalid rgba arg "[ 0.08572197  0.27759986]"
    need more than 2 values to unpack



![png](Visualization%204D_files/Visualization%204D_7_1.png)



    x = np.random.random(50)
    y = np.random.random(50)
    c = np.random.random(50)  # color of points
    s = 500 * np.random.random(50)  # size of points
    
    fig, ax = plt.subplots()
    im = ax.scatter(x, y, c=c, s=s, cmap=plt.cm.jet)
    
    # Add a colorbar
    fig.colorbar(im, ax=ax)
    
    # set the color limits - not necessary here, but good to know how.
    im.set_clim(0.6, 4.0)


![png](Visualization%204D_files/Visualization%204D_8_0.png)



    def randrange(n, vmin, vmax):
        return (vmax-vmin)*np.random.rand(n) + vmin
    def meshgrid2(*arrs):
        arrs = tuple(reversed(arrs))
        lens = map(len, arrs)
        dim = len(arrs)
        sz = 1
        for s in lens:
            sz *= s
        ans = []
        for i, arr in enumerate(arrs):
            slc = [1]*dim
            slc[i] = lens[i]
            arr2 = asarray(arr).reshape(slc)
            for j, sz in enumerate(lens):
                if j != i:
                    arr2 = arr2.repeat(sz, axis=j)
                ans.append(arr2)
        return tuple(ans)
    
    from numpy import asarray
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 5
    c = np.random.random(n)
    
    x = 30*np.random.rand(n,1)
    y = 20*np.random.rand(n,1)
    # z = 20*np.random.rand(n,1)
    G = meshgrid(x, y, z)
    x
    G[0]
    # print np.shape(g[0])
    # print np.shape(g[1])
    # print np.shape(g[2])
    sc = ax.scatter(g[2][0],g[2][1],g[2][2], c=c, s=100, marker='o', cmap = plt.cm.jet)
    # sc = ax.scatter(x,y,z, c=c, s=100, marker='o', cmap = plt.cm.jet)
    
    # for m, zl, zh in [('o', -50, -25), ('^', -30, -5)]:
    #     xs = randrange(n, 23, 32)
    #     ys = randrange(n, 0, 100)
    #     ax.scatter(xs, ys, c=c, s=100, marker=m, cmap = plt.cm.jet)
    #     zs = randrange(n, zl, zh)
    #     ax.scatter(xs, ys, zs, c=c, marker=m, cmap = plt.cm.jet)
    
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    
    im.set_clim(min(c), max(c))
    print(min(c))
    print(max(c))
    plt.colorbar(sc)
    plt.show()




    array([[[  1.60873208]],
    
           [[  5.48201339]],
    
           [[  1.43942671]],
    
           [[  5.50187473]],
    
           [[ 10.08475158]]])




![png](Visualization%204D_files/Visualization%204D_9_1.png)



    n = 10
    x = 30*np.random.rand(n,1)
    y = 20*np.random.rand(n,1)
    
    varNames = 


      File "<ipython-input-133-9b74318cb558>", line 5
        varNames = {'x'=x,'y'=y,'z'=z}
                       ^
    SyntaxError: invalid syntax




    def ndmesh(*args):
       args = map(np.asarray,args)
       return np.broadcast_arrays(*[x[(slice(None),)+(None,)*i] for i, x in enumerate(args)])
    
    n=3
    x = np.arange(0,n)#30*np.random.rand(n,1)
    y = np.arange(0,n)#20*np.random.rand(n,1)
    z = np.arange(0,n)#20*np.random.rand(n,1)
    G = ndmesh(x, y, z)
    print(np.shape(G))
    c = np.random.random(n**3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(G[0], G[1], G[2], c=c, s=100, marker='o', cmap = plt.cm.jet)
    im.set_clim(min(c), max(c))
    print(min(c))
    print(max(c))
    plt.colorbar(sc)
    plt.show()


    (3, 3, 3, 3)
    0.0484527987768
    0.994279695702



![png](Visualization%204D_files/Visualization%204D_11_1.png)



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 3
    c = np.random.random(n**2)
    
    x = 30*np.random.rand(n,1)
    y = 20*np.random.rand(n,1)
    G = np.meshgrid(x, y)
    print(np.shape(G))
    print(type(G))
    print(G[0])
    X,Y = np.meshgrid(x, y)
    print('******')
    print(X)
    
    sc = ax.scatter(X,Y, c=c, s=100, marker='o', cmap = plt.cm.jet)
    # print(np.shape(X))
    sc.set_clim(min(c), max(c))
    print(min(c))
    print(max(c))
    plt.colorbar(sc)
    plt.show()

    (2, 3, 3)
    <type 'list'>
    [[ 28.33903936   8.325851    11.31039052]
     [ 28.33903936   8.325851    11.31039052]
     [ 28.33903936   8.325851    11.31039052]]
    ******
    [[ 28.33903936   8.325851    11.31039052]
     [ 28.33903936   8.325851    11.31039052]
     [ 28.33903936   8.325851    11.31039052]]
    0.0480375273753
    0.779386792783



![png](Visualization%204D_files/Visualization%204D_12_1.png)


#### 


    


    y




    array([[  3.54267116],
           [ 16.97773363],
           [ 12.74277372],
           [ 12.33684525],
           [  8.23897565],
           [ 13.63380401],
           [  8.04782107],
           [ 18.87575884],
           [ 18.18877738],
           [ 14.03102395]])




    Y




    array([[  3.54267116,   3.54267116,   3.54267116,   3.54267116,
              3.54267116,   3.54267116,   3.54267116,   3.54267116,
              3.54267116,   3.54267116],
           [ 16.97773363,  16.97773363,  16.97773363,  16.97773363,
             16.97773363,  16.97773363,  16.97773363,  16.97773363,
             16.97773363,  16.97773363],
           [ 12.74277372,  12.74277372,  12.74277372,  12.74277372,
             12.74277372,  12.74277372,  12.74277372,  12.74277372,
             12.74277372,  12.74277372],
           [ 12.33684525,  12.33684525,  12.33684525,  12.33684525,
             12.33684525,  12.33684525,  12.33684525,  12.33684525,
             12.33684525,  12.33684525],
           [  8.23897565,   8.23897565,   8.23897565,   8.23897565,
              8.23897565,   8.23897565,   8.23897565,   8.23897565,
              8.23897565,   8.23897565],
           [ 13.63380401,  13.63380401,  13.63380401,  13.63380401,
             13.63380401,  13.63380401,  13.63380401,  13.63380401,
             13.63380401,  13.63380401],
           [  8.04782107,   8.04782107,   8.04782107,   8.04782107,
              8.04782107,   8.04782107,   8.04782107,   8.04782107,
              8.04782107,   8.04782107],
           [ 18.87575884,  18.87575884,  18.87575884,  18.87575884,
             18.87575884,  18.87575884,  18.87575884,  18.87575884,
             18.87575884,  18.87575884],
           [ 18.18877738,  18.18877738,  18.18877738,  18.18877738,
             18.18877738,  18.18877738,  18.18877738,  18.18877738,
             18.18877738,  18.18877738],
           [ 14.03102395,  14.03102395,  14.03102395,  14.03102395,
             14.03102395,  14.03102395,  14.03102395,  14.03102395,
             14.03102395,  14.03102395]])




    num_of_zeros_for_padding = 90
    pad_series_0 = np.hstack((s0_vals[0]*np.ones([num_of_zeros_for_padding,]), 3*np.ones([540]),3*np.ones([num_of_zeros_for_padding,])
        #run wavelet transformations
    wave_0 = sp.signal.cwt(pad_series_0, signal.ricker, widths=[widths])

