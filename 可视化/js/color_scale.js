var width = 210;	//画布的宽度
var height = 30;	//画布的高度

let linear = d3.scaleLinear().domain([0, 10]).range([0, 1]) //数字归一化
let compute = d3.interpolate('white', 'orange');
 
var svg = d3.select("#colorscale")				//选择文档中的body元素
			.attr("width", width)		//设定宽度
			.attr("height", height)	//设定高度

    svg.selectAll('rect').data(d3.range(100)).enter()
	.append('rect')
	.attr('x', (d,i) => i * 10)
	.attr('y', 0)
	.attr('width', 10)
	.attr('height', 30)
	.style('fill', (d,i) => compute(linear(d)))