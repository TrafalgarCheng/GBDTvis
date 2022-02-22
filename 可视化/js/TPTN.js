$.get("data/TPTN2.json", function(data){
    var dom = document.getElementById("feature_scatter");
    var myChart = echarts.init(dom);
    var app = {};
    option = null;
    var TPdata=[];
    var TNdata=[];
    var FPdata=[];
    var FNdata=[];
    for(i=0;i<data.TPx.length;i++){
        TPdata.push([data.TPx[i],data.TPy[i]])       
    }
    for(i=0;i<data.TNx.length;i++){
        TNdata.push([data.TNx[i],data.TNy[i]])       
    }
    for(i=0;i<data.FPx.length;i++){
        FPdata.push([data.FPx[i],data.FPy[i]])       
    }
    for(i=0;i<data.FNx.length;i++){
        FNdata.push([data.FNx[i],data.FNy[i]])     
    }
    // var data = [
    //     [[28604,77,17096869,'Australia',1990],[31163,77.4,27662440,'Canada',1990],[1516,68,1154605773,'China',1990],[13670,74.7,10582082,'Cuba',1990],[28599,75,4986705,'Finland',1990],[29476,77.1,56943299,'France',1990],[31476,75.4,78958237,'Germany',1990],[28666,78.1,254830,'Iceland',1990],[1777,57.7,870601776,'India',1990],[29550,79.1,122249285,'Japan',1990],[2076,67.9,20194354,'North Korea',1990],[12087,72,42972254,'South Korea',1990],[24021,75.4,3397534,'New Zealand',1990],[43296,76.8,4240375,'Norway',1990],[10088,70.8,38195258,'Poland',1990],[19349,69.6,147568552,'Russia',1990],[10670,67.3,53994605,'Turkey',1990],[26424,75.7,57110117,'United Kingdom',1990],[37062,75.4,252847810,'United States',1990]],
    //     [[44056,81.8,23968973,'Australia',2015],[43294,81.7,35939927,'Canada',2015],[13334,76.9,1376048943,'China',2015],[21291,78.5,11389562,'Cuba',2015],[38923,80.8,5503457,'Finland',2015],[37599,81.9,64395345,'France',2015],[44053,81.1,80688545,'Germany',2015],[42182,82.8,329425,'Iceland',2015],[5903,66.8,1311050527,'India',2015],[36162,83.5,126573481,'Japan',2015],[1390,71.4,25155317,'North Korea',2015],[34644,80.7,50293439,'South Korea',2015],[34186,80.6,4528526,'New Zealand',2015],[64304,81.6,5210967,'Norway',2015],[24787,77.3,38611794,'Poland',2015],[23038,73.13,143456918,'Russia',2015],[19360,76.5,78665830,'Turkey',2015],[38225,81.4,64715810,'United Kingdom',2015],[53354,79.1,321773631,'United States',2015]]
    // ];
    
    option = {
        // backgroundColor: new echarts.graphic.RadialGradient(0.3, 0.3, 0.8, [{
        //     offset: 0,
        //     color: '#f7f8fa'
        // }, {
        //     offset: 1,
        //     color: '#cdd0d5'
        // }]),
        title: {
            text: 'Prediction Performance'
        },
        grid:[ {
            height: '89%',
            width:'87%',
            top:'8%',
            left:'5%',
        }],
        legend: {
            right: 10,
            data: ['TP', 'TN','FP','FN'],
            textStyle:{
                //color:'#FFF',
            },
        },
        xAxis: {
            inverse:true,
            splitLine: {
                lineStyle: {
                    type: 'dashed'
                }
            },
            axisLine:{
               
                lineStyle:{
                    show:false,
                },
            },
            axisTick: {
                show: false
            },
            axisLabel:{
                show:false,
            }
        },
        yAxis: {
            splitLine: {
                lineStyle: {
                    type: 'dashed'
                }
            },
            axisLine:{
                onZero: false,
                lineStyle:{
                    //color:'#FFF',
                },
            },
            scale: true,
        },
        series: [{
            name: 'TP',
            data: TPdata,
            type: 'scatter',
            emphasis: {
                label: {
                    show: true,
                    textStyle:{
                        //color:'#FFF',
                    },
                    formatter: function (param) {
                        return "TP： "+param.data[0]+", "+param.data[1];
                    },
                    position: 'top'
                }
            },
            itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(3, 13, 156, 0.5)',
                shadowOffsetY: 5,
                color: new echarts.graphic.RadialGradient(0.4, 0.3, 1, [{
                    offset: 0,
                    color: 'rgb(39, 49, 194)'
                }, {
                    offset: 1,
                    color: 'rgb(78, 86, 204)'
                }])
            }
        }, {
            name: 'TN',
            data: TNdata,
            type: 'scatter',
            emphasis: {
                label: {
                    show: true,
                    textStyle:{
                        //color:'#FFF',
                    },
                    formatter: function (param) {
                        return "TN： "+param.data[0]+", "+param.data[1];;
                    },
                    position: 'top'
                }
            },
            itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(207, 79, 117, 0.5)',
                shadowOffsetY: 5,
                color: new echarts.graphic.RadialGradient(0.4, 0.3, 1, [{
                    offset: 0,
                    color: 'rgba(235, 91, 134, 0.1)'
                }, {
                    offset: 1,
                    color: 'rgb(250, 135, 170)'
                }])
            }
            
        },{
            name: 'FP',
            data: FPdata,
            type: 'scatter',
            emphasis: {
                label: {
                    show: true,
                    textStyle:{
                        //color:'#FFF',
                    },
                    formatter: function (param) {
                        return "FP： "+param.data[0]+", "+param.data[1];;
                    },
                    position: 'top'
                }
            },
            itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(25, 100, 150, 0.5)',
                shadowOffsetY: 5,
                color: new echarts.graphic.RadialGradient(0.4, 0.3, 1, [{
                    offset: 0,
                    color: 'rgb(153, 245, 255)'
                }, {
                    offset: 1,
                    color: 'rgb(25, 183, 207)'
                }])
            }
        },
        {
            name: 'FN',
            data: FNdata,
            type: 'scatter',
            emphasis: {
                label: {
                    show: true,
                    textStyle:{
                        //color:'#FFF',
                    },
                    formatter: function (param) {
                        return "FN： "+param.data[0]+", "+param.data[1];;
                    },
                    position: 'top'
                }
            },
            itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(255, 255, 0, 0.5)',
                shadowOffsetY: 5,
                color: new echarts.graphic.RadialGradient(0.4, 0.3, 1, [{
                    offset: 0,
                    color: 'rgb(240, 240, 120)'
                }, {
                    offset: 1,
                    color: 'rgb(252, 252, 200)'
                }])
            }
        }
    ]
    };
    if (option && typeof option === "object") {
        myChart.setOption(option, true);
    }
})