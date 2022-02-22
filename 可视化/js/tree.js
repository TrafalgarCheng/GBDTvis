var id=0;
var maxId=0;
var treenum=0;
var maxValue=-10000;
var minValue=10000;
var node=[]
var edge=[]
var isRoot=[]
var isLeftTree=[]
var path;
var matrix=[];
var g;

var dir=['TB','LR'];
var btnswitch=0;
var cellheight=[70,100];
var cellwidth=[130,150];
function traverse(obj,flag){
    tipcolor="#87CEFA";
    if(obj.label.includes("leaf")){
        var word=obj.label.split(" ");
        if(eval(minValue)>eval(word[2])){
            minValue=word[2];
        }
        if(maxValue==-10000){
            maxValue=word[2];
            maxId=id;
        }
        else{
            if(eval(maxValue)<eval(word[2])){
                maxValue=word[2];
                maxId=id;
            }
        }
        NODE={
            id: id,
            label: obj.label,
            shape: "ellipse",
            name:obj.id,
            color:tipcolor
        }
    }
    else{
        NODE={
            id: id,
            label: obj.label,
            shape: "ellipse",
            name:obj.id,
            color:"#87CEFA"
        }
    }
    
    node.push(NODE);
    if(!obj.children){
        return 
    }
    cnt=0;
    for(var i in obj.children){ 
        id=id+1;
        if(cnt==0){
            isLeftTree[flag]=id;
            cnt++;
        }
        EDGE={
               start: flag, end: id, option: {}
        }
        if(matrix[flag]==undefined){
            matrix[flag]=[id]
        }
        else{
            matrix[flag].push(id)
        }
        edge.push(EDGE);
        traverse(obj.children[i],id)
    }
};

function dfs(id,str){
    if(id==maxId){
        path=str+" "+maxId;
        return;
    }
    else{
        for(var i in matrix[id]){
            dfs(matrix[id][i],str+" "+matrix[id][i]);
        }
    }
}

// function getMatrix(){
//     for(var i in edge){
//         console.log("start:"+edge[i].start)
//         if(matrix[edge[i].start]==undefined){
//             console.log(edge[i].start+"undefined")
//             matrix[edge[i].start]=[edge[i].end]
//             console.log(" edge[i].start:"+matrix[edge[i].start])
//         }
//         else{
//             matrix[edge[i].start].push(edge[i].end);
//             console.log("defined edge[i].start:"+matrix[edge[i].start])
//         }
//     }
// }


$.get("data/Tree1.json", function(data){
    traverse(data,id);
    //console.log(id+":"+maxId);
    //console.log(edge);
    //console.log("maxid: "+maxId);
    dfs(0,"0");
    //console.log(path)
    switchController();
    render();
})

function switchController(){
    var Main = {
        methods:{
          onSwitch() {
            btnswitch=(btnswitch+1)%2;
            render();
        },
      }
    }
    
    var Ctor = Vue.extend(Main)
    new Ctor().$mount('#switch');

}


function render(){
    console.log("click")
    g = new dagreD3.graphlib.Graph()
    .setGraph({
        rankdir:dir[btnswitch]
    })
    .setDefaultEdgeLabel(function () { return {}; });
    for (let i in node) { //画点
        
        let el = node[i]
        if(el.id==0){           //根节点
            g.setNode(i, {
                width:cellwidth[btnswitch], //节点长度
                height:cellheight[btnswitch],//结点宽度
                id: el.id,
                shape:el.shape,
                label: el.label,
                style: "fill:"+"#FFFFFF"+";stroke:#333;stroke-width:1.5px"//节点样式
            });
        }
        else if(el.label.includes("leaf")){   //如果为叶子结点
            var word=el.label.split(" ")
            if(treenum==0)                      //树不同 透明度计算方法不一样
                a=(word[2]-minValue)/(maxValue-minValue)* 0.35;
            else if(treenum==1)
                a=0.4+(word[2]-minValue)/(maxValue-minValue)* 0.35;
            else
                a=0.65+(word[2]-minValue)/(maxValue-minValue)* 0.35;

            //console.log(maxValue);
            g.setNode(i, {
                width:cellwidth[btnswitch], //节点长度
                height:cellheight[btnswitch],//结点宽度
                id: el.id,
                shape:el.shape,
                label: el.label,
                labelStyle: "font-weight: bold;font-size: 1.3em",
                style: "fill:"+"rgba(255,69,0,"+a+")"+";stroke:#333;stroke-width:1.5px"//节点样式
            });
        }
        else {   //如果是节点
            var word=el.label.split(" ")
            if(treenum==0)                      //树不同 透明度计算方法不一样
                a=(word[3]-minValue)/(maxValue-minValue)* 0.35;
            else if(treenum==1)
                a=0.4+(word[3]-minValue)/(maxValue-minValue)* 0.35;
            else
                a=0.65+(word[3]-minValue)/(maxValue-minValue)* 0.35;
            g.setNode(i, {
                width:cellwidth[btnswitch], //节点长度
                height:cellheight[btnswitch],//结点宽度
                id: el.id,
                shape:el.shape,
                label: el.label,
                labelStyle: "font-weight: bold",
                style: "fill:"+"rgba(255,69,0,"+a+")"+";stroke:#333;stroke-width:1.5px"//节点样式
            });
        }
        // else{
        //     g.setNode(i, {
        //         width:cellwidth[btnswitch], //节点长度
        //         height:cellheight[btnswitch],//结点宽度
        //         id: el.id,
        //         shape:el.shape,
        //         label: el.label,
        //         style: "fill:"+"#FFFF00"+";stroke:#333;stroke-width:1.5px"//节点样式
        //     });
        // }
    }
    
    for (let i in edge) { // 画连线
        let el = edge[i]
        flag=0
        var pathword=path.split(" ")
        for(var j=0; j<pathword.length-1;j++){
            if(el.start==pathword[j] && el.end==pathword[j+1]){
                flag=1;
            }
        }
        if(flag==1){    //为最大值路径
            g.setEdge(el.start, el.end, {
                style: "stroke: #000; fill: none;stroke-width:5",
                arrowheadStyle: "fill: #000;stroke: #fff;",
                arrowheadClass: 'arrowhead',
                arrowhead: 'vee',
            });
        }
        else{
            if(isLeftTree[el.start]==el.end){
                g.setEdge(el.start, el.end, {
                    style: "stroke: #0000FF; fill: none;stroke-width:2",
                    arrowheadStyle: "fill: #0000FF;stroke: #0000FF;",
                    //arrowheadClass: 'arrowhead',
                    arrowhead: 'normal',
                    curve: d3.curveBasis
                });
            }
            else{
                g.setEdge(el.start, el.end, {
                    style: "stroke: #FF0000; fill: none;stroke-width:2",
                    arrowheadStyle: "fill: #FF0000;stroke: #FF0000",
                    //arrowheadClass: 'arrowhead',
                    arrowhead: 'normal',
                    curve: d3.curveBasis
                });
            }
            
        }

    }
    var render = new dagreD3.render();
    var svg = d3.select("#dataflow"); //声明节点
    svg.select("g").remove(); //删除以前的节点，清空画面
    var svgGroup = svg.append("g");
    var inner = svg.select("g");
    var zoom = d3.zoom().on("zoom", function () { //添加鼠标滚轮放大缩小事件
      inner.attr("transform", d3.event.transform);
    });
    svg.call(zoom);
    render(d3.select("svg g"), g); //渲染节点
    let max = svg._groups[0][0].clientWidth>svg._groups[0][0].clientHeight?svg._groups[0][0].clientWidth:svg._groups[0][0].clientHeight;
    var initialScale = max/779; //initialScale元素放大倍数，随着父元素宽高发生变化时改变初始渲染大小
    var tWidth = (svg._groups[0][0].clientWidth  - g.graph().width * initialScale) / 2; //水平居中
    var tHeight = (svg._groups[0][0].clientHeight  - g.graph().height * initialScale) / 2; //垂直居中
    svg.call(zoom.transform, d3.zoomIdentity.translate(tWidth, tHeight).scale(initialScale)); //元素水平垂直居中

}
