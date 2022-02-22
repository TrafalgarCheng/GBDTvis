var Main = {
    data() {
        return {
          tree_options: [{
            value: '选项1',
            label: 'Tree1'
          }, {
            value: '选项2',
            label: 'Tree50'
          }, {
            value: '选项3',
            label: 'Tree100'
          }],
          tree_value: ''
        }
      }
  }
var Ctor = Vue.extend(Main)
new Ctor().$mount('#tree_selection')
