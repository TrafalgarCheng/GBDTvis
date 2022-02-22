$.get("data/pdp/C14.json", function(data){
    var dom = document.getElementById("c14_charts");
    var myChart = echarts.init(dom);
    var app = {};
    var xdataAxis = data.bins;
    var ydataAxis = data.hist;;
    var pdpx=data.pdpx;
    var pdpy=data.pdpy;
    var dataAxis = [];
    var minData=Math.floor(Math.min(xdataAxis));
    var maxData=Math.ceil(Math.max(xdataAxis));
    var intervalNum=2;

    option = null;
    option={
        title:{
            text: 'C14  (403)', //主标题文本，'\n'指定换行
            x:'center',
            //subtext: 'Value: 403',
            textStyle:{
                color:'#000',
                fontFamily:'Verdana'
            }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {            // 坐标轴指示器，坐标轴触发有效
                type: 'shadow'        // 默认为直线，可选为：'line' | 'shadow'
            },
            label: {
                show: true
            }
        },
        grid:[ {
            height: '50%',
            width:'84%',
            top:'1%'
        },
        {
            height: '30%',
            width:'84%',
            top:'66%'
        },
        ],
        xAxis: [{
                type: 'category',
                data: pdpx,
                axisLabel: {
                    show:true,
                    textStyle: {
                        lineHeight: 1, //行高
                    },
                },
                axisTick: {
                    show: false
                },
                axisLine: {
                    show: true,
                    onZero: false
                },
                splitLine: {
                    show: false     //取消网格线
                },
                boundaryGap: false,
                gridIndex:0,
            },
            {
                type: 'category',
                position: 'top',
                data: xdataAxis,
                axisLabel: {
                    //interval:0,
                    show:false,
                    // inside: false,
                    textStyle: {
                         color: '#999'
                     },
                    // fontSize:20
                    textStyle: {
                        lineHeight: 1, //行高
                    },
                },
                axisTick: {
                    show: false
                },
                axisLine:{
                    lineStyle:{
                        color:'#000',
                    },
                },
                splitLine: {
                    show: false     //取消网格线
                },
                z: 10,
                gridIndex:1,
            },
            
        ],
        yAxis: [{
            type: 'value',
            splitLine: {
                show: false     //取消网格线
            },
            axisLabel: {
                show:true,
            },
            axisTick: {
                show: true
            },
            axisLine:{
                lineStyle:{
                    color:'#000',
                },
            },
            gridIndex:0,
        },
        {
            type:'value',
            axisLine:{
                lineStyle:{
                    color:'#000',
                },
            },
            axisTick: {
                show: false
            },
            axisLabel: {
                show:true,
                textStyle: {
                    fontSize:8
                }
            },
            splitLine: {
                show: false     //取消网格线
            },
            z: 10,
            inverse: true,
            gridIndex:1,
        },
    ],
    series: [{
            name: 'Pdp',
            data: pdpy,
            type: 'line',
            color:'#fa2c7b',
            smooth: true,
            xAxisIndex:0,
            yAxisIndex:0,
        },
        {
            name: 'Zero',
            data: [0,0,0,0,0,0,0,0,0,0],
            itemStyle:{
                normal:{
                    lineStyle:{
                        width:1,
                        type:'dotted' //设置线条为虚线
                    }
                }
            },
            type: 'line',
            color:'#000',
            smooth: true,
            xAxisIndex:0,
            yAxisIndex:0,
            symbol:'none'    //去掉折线上的小圆点
        },
        {
            name: 'Histogram',
            type: 'bar',
                name:'bins',
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(
                        0, 0, 0, 1,
                        [
                            {offset: 0, color: '#fa2c7b'},
                            {offset: 0.5, color: '#fa2c7b'},
                            {offset: 1, color: '#fa2c7b'}
                        ]
                    )
                },
                data: ydataAxis,
                xAxisIndex:1,
                yAxisIndex:1,
        },
        ]
    };
    if (option && typeof option === "object") {
        myChart.setOption(option, true);
    }
})