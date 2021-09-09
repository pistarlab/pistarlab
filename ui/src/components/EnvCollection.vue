<template>
<div v-if="collection && envs">
    <b-card class="h-100 card-shadow card-flyer" no-body header-bg-variant="white" header-text-variant="dark">
        <!-- <div class="text-center image-box">
                       <div class="d-flex flex-wrap"  style="max-width:200px">
            <div v-for="(item,idx) in envs" v-bind:key="idx" >
             <img :src="`${appConfig.API_URL}/api/env_preview_image/env_${item.ident}`" alt="No Image Found" style="max-height:30px;"/> 
            </div>
            </div>
        </div> -->
        <b-card-body >
            <h4> {{ collection }}</h4>

        </b-card-body>
    </b-card>
</div>
</template>

<script>
import {
    appConfig
} from "../app.config";
export default {
    props: {
        collection:String,
        envs: Array
    },
    data() {
        return {
            appConfig

        };
    },
    mounted() {
        //
    },
    methods: {
        //

    },
    computed: {
        //

    },
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style>

</style>
