from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import tqdm
import scanpy as sc
import networkx as nx
import numpy
import operator
import collections
import os 
import seaborn as sns

class GeneEmbedding(object):

    """
    This class provides an interface to the gene embedding, which can be used for tasks such as similarity computation, visualization, etc.

    :param embedding_file: Specifies the path to a set of .vec files generated for model training.
    :type embedding_file: str
    :param dataset: The GeneVectorDataset object that was constructed from the original AnnData object.
    :type dataset: :class:'genevector.dataset.GeneVectorDataset'
    :param vector: Specifies if using the first set of weights ("1"), the second set of weights ("2"), or the average ("average").
    :type vector: str
    """

    def __init__(self, embedding_file, vector="average"):
        """Constructor method
        """
        if vector not in ("1","2","average"):
            raise ValueError("Select the weight vector from: ('1','2','average')")
        if vector == "average":
            print("Loading average of 1st and 2nd weights.")
            avg_embedding = embedding_file.replace(".vec","_avg.vec")
            secondary_weights = embedding_file.replace(".vec","2.vec")
            GeneEmbedding.average_vector_results(embedding_file,secondary_weights,avg_embedding)
            self.embeddings = self.read_embedding(avg_embedding)
        elif vector == "1":
            print("Loading first weights.")
            self.embeddings = self.read_embedding(embedding_file)
        elif vector == "2":
            print("Loading second weights.")
            secondary_weights = embedding_file.replace(".vec","2.vec")
            self.embeddings = self.read_embedding(secondary_weights)
        self.vector = []
        self.embedding_file = embedding_file
        self.vector = []
        self.genes = []
        for gene in tqdm.tqdm(self.embeddings.keys()):
            self.vector.append(self.embeddings[gene])
            self.genes.append(gene)

    def read_embedding(self, filename):
        embedding = dict()
        lines = open(filename,"r").read().splitlines()[1:]
        for line in lines:
            vector = line.split()
            gene = vector.pop(0)
            embedding[gene] = numpy.array([float(x) for x in vector])
        return embedding

    def get_adata(self, resolution=20.):
        """
        This method returns the AnnData object that contains the gene embedding with leiden clusters for metagenes, the neighbors graph, and the UMAP embedding.

        :param resolution: The resolution to pass to the sc.tl.leiden function.
        :type resolution: float
        :return: An AnnData object with metagenes stored in 'leiden' for the provided resolution, the neighbors graph, and UMAP embedding.
        :rtype: AnnData
        """

        mat = numpy.array(self.vector)
        numpy.savetxt(".tmp.txt",mat)
        gdata = sc.read_text(".tmp.txt")
        os.remove(".tmp.txt")
        gdata.obs.index = self.genes
        sc.pp.neighbors(gdata,use_rep="X")
        sc.tl.leiden(gdata,resolution=resolution)
        sc.tl.umap(gdata)
        return gdata

    def plot_similarities(self, gene, n_genes=10, save=None):
        """
        Plot a horizontal bar plot of cosine similarity of the most similar vectors to 'gene' argument.

        :param gene: The gene symbol of the gene of interest.
        :type gene: str
        :param save: The path to save the figure (optional).
        :type gene: str, optional
        :return: A matplotlib axes object representing the plot.
        :rtype:  matplotlib.figure.axes
        """
        df = self.compute_similarities(gene).head(n_genes)
        _,ax = plt.subplots(1,1,figsize=(3,6))
        sns.barplot(data=df,y="Gene",x="Similarity",palette="magma_r",ax=ax)
        ax.set_title("{} Similarity".format(gene))
        if save != None:
            plt.savefig(save)
        return ax

    def plot_metagene(self, gdata, mg=None, title="Gene Embedding"):
        """
        Plot a UMAP with the genes from a given metagene highlighted and annotated.

        :param gdata: The AnnData object holding the gene embedding (from embedding.get_adata).
        :type gdata: AnnData
        :param mg: The metagene identifier (leiden cluster number) (optional).
        :type mg: str, optional
        :param title: The title of the plot. (optional).
        :type title: str, optional
        """
        highlight = []
        labels = []
        clusters = collections.defaultdict(list)
        for x,y in zip(gdata.obs["leiden"],gdata.obs.index):
            clusters[x].append(y)
            if x == mg:
                highlight.append(str(x))
                labels.append(y)
            else:
                highlight.append("_Other")
        _labels = []
        for gene in labels:
            _labels.append(gene)
        gdata.obs["Metagene {}".format(mg)] = highlight
        _,ax = plt.subplots(1,1,figsize=(8,6))
        sc.pl.umap(gdata,alpha=0.5,show=False,size=100,ax=ax)
        sub = gdata[gdata.obs["Metagene {}".format(mg)]!="_Other"]
        sc.pl.umap(sub,color="Metagene {}".format(mg),title=title,size=200,show=False,add_outline=False,ax=ax)
        for gene, pos in zip(gdata.obs.index,gdata.obsm["X_umap"].tolist()):
            if gene in _labels:
                ax.text(pos[0]+.04, pos[1], str(gene), fontsize=6, alpha=0.9, fontweight="bold")
        plt.tight_layout()

    def get_vector(self, gene):
        return self.embeddings[gene]

    def plot_metagenes_scores(self, adata, metagenes, column, plot=None):
        """
        Plot a Seaborn clustermap with the gene module scores for a list of metagenes over a covariate (column). Requires running score_metagenes previously.

        :param adata: The AnnData object holding the cell embedding (from embedding.CellEmbedding.get_adata).
        :type adata: AnnData
        :param metagenes: Dict of metagenes identifiers to plot in clustermap.
        :type metagenes: dict
        :param column: Covariate in obs dataframe of AnnData.
        :type column: str
        :param column: Covariate in obs dataframe of AnnData.
        :type column: str
        :param plot: Filename for saving a figure.
        :type plot: str
        """
        plt.figure(figsize = (5, 13))
        matrix = []
        meta_genes = []
        cfnum = 1
        cfams = dict()
        for cluster, vector in metagenes.items():
            row = []
            cts = []
            for ct in set(adata.obs[column]):
                sub = adata[adata.obs[column]==ct]
                val = numpy.mean(sub.obs[str(cluster)+"_SCORE"].tolist())
                row.append(val)
                cts.append(ct)
            matrix.append(row)
            label = str(cluster)+"_SCORE: " + ", ".join(vector[:10])
            if len(set(vector)) > 10:
                label += "*"
            meta_genes.append(label)
            cfams[cluster] = label
            cfnum+=1
        matrix = numpy.array(matrix)
        df = pandas.DataFrame(matrix,index=meta_genes,columns=cts)
        plt.figure()
        sns.clustermap(df,figsize=(5,9), dendrogram_ratio=0.1,cmap="mako",yticklabels=True, standard_scale=0)
        plt.tight_layout()
        if plot:
            plt.savefig(plot)

    def score_metagenes(self,adata ,metagenes):
        """
        Score a list of metagenes (get_metagenes) over all cells. 

        :param adata: The AnnData object holding the cell embedding (from embedding.CellEmbedding.get_adata).
        :type adata: AnnData
        :param metagenes: Dict of metagenes identifiers to plot in clustermap.
        :type metagenes: dict
        """
        for p, genes in metagenes.items():
            try:
                sc.tl.score_genes(adata,score_name=str(p)+"_SCORE",gene_list=genes)
                scores = numpy.array(adata.obs[str(p)+"_SCORE"].tolist()).reshape(-1,1)
                scaler = MinMaxScaler()
                scores = scaler.fit_transform(scores)
                scores = list(scores.reshape(1,-1))[0]
                adata.obs[str(p)+"_SCORE"] = scores
            except Exception as e:
                adata.obs[str(p)+"_SCORE"] = 0.
            

    def get_metagenes(self, gdata):
        """
        Score a list of metagenes (get_metagenes) over all cells. 

        :param gdata: The AnnData object holding the gene embedding (from embedding.GeneEmbedding.get_adata).
        :type gdata: AnnData
        :return: A dictionary of metagenes (identifier, gene list).
        :rtype:  dict
        """
        metagenes = collections.defaultdict(list)
        for x,y in zip(gdata.obs["leiden"],gdata.obs.index):
            metagenes[x].append(y)
        return metagenes

    def compute_similarities(self, gene, subset=None):
        """
        Compute the cosine similarities between a target gene and all other vectors in the embedding.

        :param gene: Target gene to compute cosine similarities.
        :type gene: str
        :param subset: Only compute against a subset of gene vectors. (optional).
        :type subset: list, optional
        :return: A pandas dataframe holding a gene symbol column ("Gene") and a cosine similarity column ("Similarity").
        :rtype:  pandas.DataFrmae
        """
        if gene not in self.embeddings:
            return None
        embedding = self.embeddings[gene]
        distances = dict()
        if subset:
            targets = set(list(self.embeddings.keys())).intersection(set(subset))
        else:
            targets = list(self.embeddings.keys())
        for target in targets:
            if target not in self.embeddings:
                continue
            v = self.embeddings[target]
            distance = float(cosine_similarity(numpy.array(embedding).reshape(1, -1),numpy.array(v).reshape(1, -1))[0])
            distances[target] = distance
        sorted_distances = list(reversed(sorted(distances.items(), key=operator.itemgetter(1))))
        genes = [x[0] for x in sorted_distances]
        distance = [x[1] for x in sorted_distances]
        df = pandas.DataFrame.from_dict({"Gene":genes, "Similarity":distance})
        return df

    def generate_weighted_vector(self, genes, markers, weights):
        vector = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes and gene in weights:
                vector.append(weights[gene] * numpy.array(vec))
            if gene not in genes and gene in markers and gene in weights:
                vector.append(list(weights[gene] * numpy.negative(numpy.array(vec))))
        return list(numpy.sum(vector, axis=0))


    def generate_vector(self, genes):
        """
        Compute an averagve vector representation for a set of genes in the learned gene embedding.

        :param genes: List of genes to generate an average vector embedding.
        :type genes: list
        :return: The average vector for a set of genes in the gene embedding.
        :rtype:  list
        """
        vector = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes:
                vector.append(vec)
        assert len(vector) != 0, genes
        return list(numpy.average(vector, axis=0))

    def generate_weighted_vector(self, genes, weights):
        """
        Compute an averagve vector representation for a set of genes in the learned gene embedding with a set of weights.

        :param genes: List of genes to generate an average vector embedding.
        :type genes: list
        :param weights: List of floats in the same order of genes to weight each vector.
        :type genes: list
        :return: The average vector for a set of genes in the gene embedding.
        :rtype:  list
        """
        vector = []
        weight = []
        for gene, vec in zip(self.genes, self.vector):
            if gene in genes and gene in weights:
                vector.append(vec)
                weight.append(weights[gene])
        assert len(vector) != 0, genes
        return list(numpy.average(vector, axis=0, weights=weight))


    def cluster_definitions_as_df(self, top_n=20):
        similarities = self.cluster_definitions
        clusters = []
        symbols = []
        for key, genes in similarities.items():
            clusters.append(key)
            symbols.append(", ".join(genes[:top_n]))
        df = pandas.DataFrame.from_dict({"Cluster Name":clusters, "Top Genes":symbols})
        return df

    @staticmethod
    def read_vector(vec):
        lines = open(vec,"r").read().splitlines()
        dims = lines.pop(0)
        vecs = dict()
        for line in lines:
            try:
                line = line.split()
                gene = line.pop(0)
                vecs[gene] = list(map(float,line))
            except Exception as e:
                continue
        return vecs, dims

    def get_similar_genes(self, vector):
        """
        Computes the similarity of each gene in the mebedding to a target vector representation.

        :param vector: Vector representation used to find the gene similarity by cosine cosine.
        :type genes: list or numpy.array
        :return: A pandas dataframe holding the gene symbol column ("Gene") and a cosine similarity column ("Similarity").
        :rtype:  pandas.DataFrmae
        """
        distances = dict()
        targets = list(self.embeddings.keys())
        for target in targets:
            if target not in self.embeddings:
                continue
            v = self.embeddings[target]
            distance = float(cosine_similarity(numpy.array(vector).reshape(1, -1),numpy.array(v).reshape(1, -1))[0])
            distances[target] = distance
        sorted_distances = list(reversed(sorted(distances.items(), key=operator.itemgetter(1))))
        genes = [x[0] for x in sorted_distances]
        distance = [x[1] for x in sorted_distances]
        df = pandas.DataFrame.from_dict({"Gene":genes, "Similarity":distance})
        return df

    def generate_network(self, threshold=0.5):
        """
        Computes networkx graph representation of the gene embedding.

        :param threshold: Minimum cosine similarity to includea as edge in the graph.
        :type genes: float
        :return: A networkx graph with each gene as a node and the edges weighted by cosine similarity.
        :rtype:  networkx.Graph
        """
        G = nx.Graph()
        a = pandas.DataFrame.from_dict(self.embeddings).to_numpy()
        similarities = cosine_similarity(a.T)
        genes = list(self.embeddings.keys())
        similarities[similarities < threshold] = 0
        edges = []
        nz = list(zip(*similarities.nonzero()))
        for n in tqdm.tqdm(nz):
            edges.append((genes[n[0]],genes[n[1]]))
        G.add_nodes_from(genes)
        G.add_edges_from(edges)
        return G

    @staticmethod
    def average_vector_results(vec1, vec2, fname):
        output = open(fname,"w")
        vec1, dims = GeneEmbedding.read_vector(vec1)
        vec2, _ = GeneEmbedding.read_vector(vec2)
        genes = list(vec1.keys())
        output.write(dims+"\n")
        for gene in genes:
            v1 = vec1[gene]
            v2 = vec2[gene]
            meanv = []
            for x,y in zip(v1,v2):
                meanv.append(str((x+y)/2))
            output.write("{} {}\n".format(gene," ".join(meanv)))
        output.close()

