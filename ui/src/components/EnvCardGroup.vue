<template>
<div v-if="item.ident">
    <b-card class="h-100 card-shadow card-flyer" no-body header-bg-variant="white" header-text-variant="dark">

        <div class="text-center image-box">
            <b-card-img :src="`${appConfig.API_URL}/api/env_preview_image/${item.ident}`" alt="No Image Found" style="max-height:200px;"></b-card-img>
        </div>

        <b-card-text class="ml-3 mb-2 mt-3 h-100 flex-column d-flex">
            <h4> {{ item.displayedName }}</h4>
            <div >
                <span class="data_label  mt-2">Categories: </span>
                <span >{{ item.categories }}</span>
            </div>
            <div v-if="item.specs">
                <span class="data_label  mt-2">Specs: </span>
                <span >{{ item.specs.length }}</span>
            </div>
        </b-card-text>

    </b-card>
</div>
</template>

<script>
import {
    appConfig
} from "../app.config";
export default {
    props: {
        item: Object
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
