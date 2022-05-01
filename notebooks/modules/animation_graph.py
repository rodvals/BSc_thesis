# functions
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_init():
    return

def replot(count, vt_size, t_span, network, ax):
    count = int(count)
    colors  = []
    for i in range(len(vt_size[count])):
        if vt_size[count, i] > 0:
            colors.append('magenta')
        elif vt_size[count, i] < 0:
            colors.append('cyan')
    # take the size of the vertex from the x coordinate
    dots = ax.get_children()[0]
    dots.set_color(colors)
    dots.set_sizes(np.abs(vt_size[count]))
#    layout = "fruchterman_reingold"

def graph_plot(v_size, t_span, network, ax):
    layout = "grid"
    ig.plot(network, target = ax, margin = (200, 200, 200, 200),
#            layout=layout,        
            edge_width=1,
            edge_arrow_size = 1,
            vertex_size = 1 ,
            vertex_color=['red'], 
            vertex_frame_color='black',
            vertex_frame_width=1000,
            edge_color = 'black',
           # vertex_shape = 'triangle',
            keep_aspect_ratio=True
#            vertex_label=['first', 'second', 'third', 'fourth'],
#            edge_width=[1, 4],
#            edge_color=['black', 'grey'],
            )   
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    

def plot_xyz_graph()
	# name_graph = "Watts Strogatz : n = {} k = {} p = {}".format(N, k, p)
	name_graph = "Erdos Renyi : n = {} p = {}".format(N, k, p)
	graph = ig.Graph.Erdos_Renyi(n=N, p=p, directed=False, loops=False)
	# g_watts_strogatz = ig.Graph.Watts_Strogatz(dim = 1, size = 20, nei = 1, p = 0.2)
	# g_barabasi_albert = ig.Graph.Barabasi(n = N, m = 2 )
	ig.plot(graph, layout='grid')
	adj = graph.get_adjacency()
	plt.show()

	# integrate dynamics 
	sol = odeint(func= rossler, y0 = q_0, t=t_span, args=(a, b, c, r, sigma, N, adj) )

	# plot coordinates
	plt.close('all')
	#new figure to plot network graphs
	fig = plt.figure(figsize = (9, 9) )
	fig.suptitle("Rossler Oscillator N = {5} with parameters: \n a = {0:.6g} b = {1:.6g} c = {2:.6g} r = {3:.6g} sigma = {4:.6g}".format(a, b, c, r, sigma, N))
	axs_xy = fig.add_subplot(2,2,1)
	axs_xy.set(xlabel="x", ylabel="y", title= "XY plot")
	plt.plot(sol[:,0::3][::100], sol[:,1::3][::100])

	axs_network = fig.add_subplot(2,2,2)
	ig.plot(graph, target = axs_network)
	#axs_network.set(title="Erdos-Renyi: p = {}, N = {}".format(p, N))
	axs_network.set(title=name_graph)

	axs_xz = fig.add_subplot(2, 2, 3)
	plt.plot(sol[:,0::3][::100], sol[:,2::3][::100])
	axs_xz.set(xlabel="x", ylabel="z", title= "XZ plot")

	axs_yz = fig.add_subplot(2, 2, 4)
	plt.plot(sol[:,1::3][::100], sol[:,2::3][::100])
	axs_yz.set(xlabel="y", ylabel="z", title= "YZ plot")

	plt.show()

	# plot time vs axis
	fig = plt.figure(figsize = (9, 9) )
	fig.suptitle("Rossler Oscillator N = {5} with parameters: \n a = {0:.6g} b = {1:.6g} c = {2:.6g} r = {3:.6g} sigma = {4:.6g}".format(a, b, c, r, sigma, N))
	axs_xt = fig.add_subplot(2,2,1)
	axs_xt.set(xlabel="x", ylabel="t", title= "t X plot")
	plt.plot(t_span[::100], sol[:,0::3][::100])

	axs_network = fig.add_subplot(2,2,2)
	ig.plot(graph, target = axs_network)
	axs_network.set(title=name_graph)

	axs_yt = fig.add_subplot(2, 2, 3)
	plt.plot(t_span[::100], sol[:,1::3][::100])
	axs_yt.set(xlabel="t", ylabel="y", title= "tY plot")

	axs_zt = fig.add_subplot(2, 2, 4)
	plt.plot(t_span[::100], sol[:,2::3][::100])
	axs_zt.set(xlabel="t", ylabel="z", title= "tZ plot")

	plt.show()


