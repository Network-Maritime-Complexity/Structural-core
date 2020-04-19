function plot_figure_TEU

% load data
load('trade_GLSN.mat', 'x')
load('trade_GLSN_metadata.mat', 'TEU')

% embedding
coords = coalescent_embedding(x, 'RA2', 'ISO', 'EA', 2);

% plot figure
figure('color', 'white')
TEU_RGB = plot_embedding_TEU(x, coords, TEU);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function TEU_RGB = plot_embedding_TEU(x, coords, TEU)

prec = 1000;
[TEU_new, colors_map] = map_TEU(TEU, prec);
colormap(colors_map);

[~,idx] = sort(TEU_new, 'descend');

hold on
radius = max(coords(:,2));
[coords_x,coords_y] = pol2cart(coords(:,1),coords(:,2));
[h1,h2] = gplot(x, [coords_x, coords_y], 'k');
plot(h1, h2, 'Color', [0.7,0.7,0.7], 'LineWidth', 0.001)
scatter(coords_x(idx), coords_y(idx), 300, TEU_new(idx), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);

xlim([-radius,radius])
ylim([-radius,radius])
axis square
axis off

TEU_RGB = colors_map(TEU_new,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TEU_new, colors_map] = map_TEU(TEU, prec)
TEU_new = log10(TEU);
TEU_new = round(TEU_new * prec);
TEU_new_min = min(TEU_new);
TEU_new = TEU_new - TEU_new_min + 1;

colors_map = colormap_blue_to_red(max(TEU_new));

TEU_ticklabels = ceil(min(log10(TEU))):floor(max(log10(TEU)));
temp = cell(size(TEU_ticklabels));
for i = 1:length(TEU_ticklabels)
    temp{i} = ['10^' num2str(TEU_ticklabels(i))];
end
%TEU_ticklabels = temp; clear temp;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function colors = colormap_blue_to_red(n)

colors = zeros(n,3);
m = round(linspace(1,n,4));
colors(1:m(2),2) = linspace(0,1,m(2));
colors(1:m(2),3) = 1;
colors(m(2):m(3),1) = linspace(0,1,m(3)-m(2)+1);
colors(m(2):m(3),2) = 1;
colors(m(2):m(3),3) = linspace(1,0,m(3)-m(2)+1);
colors(m(3):n,1) = 1;
colors(m(3):n,2) = linspace(1,0,n-m(3)+1);
